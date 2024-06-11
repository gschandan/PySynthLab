import random
import typing
from typing import List, Dict, Tuple, Set, Callable, Collection

import pyparsing
import dataclasses

from z3 import *
from z3 import ExprRef, FuncDeclRef, QuantifierRef

from src.helpers.parser.src import ast
from src.helpers.parser.src.ast import Program, CommandKind
from src.helpers.parser.src.resolution import FunctionKind, SortDescriptor, FunctionDescriptor
from src.helpers.parser.src.symbol_table_builder import SymbolTableBuilder
from src.helpers.parser.src.v1.parser import SygusV1Parser
from src.helpers.parser.src.v1.printer import SygusV1ASTPrinter
from src.helpers.parser.src.v2.parser import SygusV2Parser
from src.helpers.parser.src.v2.printer import SygusV2ASTPrinter


@dataclasses.dataclass
class SynthesisProblemOptions:
    options: object = dataclasses.field(default_factory=dict)
    sygus_standard: int = 1
    verbose: int = 1


@dataclasses.dataclass
class SynthesisProblemContext:
    enumerator_solver: Solver = dataclasses.field(default_factory=Solver)
    verification_solver: Solver = dataclasses.field(default_factory=Solver)
    original_assertions: List[ExprRef] = dataclasses.field(default_factory=list)
    constraints: List[ast.Command] = dataclasses.field(default_factory=list)
    z3_variables: Dict[str, ExprRef] = dataclasses.field(default_factory=dict)
    z3_synth_functions: Dict[str, FuncDeclRef] = dataclasses.field(default_factory=dict)
    z3_synth_function_args: Dict[str, Dict[str, ExprRef]] = dataclasses.field(default_factory=dict)
    z3_predefined_functions: Dict[str, FuncDeclRef] = dataclasses.field(default_factory=dict)
    z3_constraints: List[ExprRef] = dataclasses.field(default_factory=list)
    assertions: Set[ExprRef] = dataclasses.field(default_factory=set)
    counterexamples: List[Tuple[Dict[str, ExprRef], ExprRef]] = dataclasses.field(default_factory=list)
    negated_constraints: Set[ExprRef] = dataclasses.field(default_factory=set)
    additional_constraints: List[ExprRef] = dataclasses.field(default_factory=list)
    synth_functions: List[Dict[str, typing.Union[str, List[ExprRef], SortRef]]] = dataclasses.field(
        default_factory=list)
    smt_problem: str = ""


class SynthesisProblem:
    """
    A class representing a synthesis problem in the SyGuS format.
    """
    MIN_CONST = -2
    MAX_CONST = 2
    pyparsing.ParserElement.enablePackrat()

    def __init__(self, problem: str, options: object = None):
        """
        Initialize a SynthesisProblem instance.

        :param problem: The input problem in the SyGuS format.
        :param options: Additional options (default: None).
        """
        if options is None:
            options = SynthesisProblemOptions()

        self.input_problem: str = problem
        self.options = options
        self.parser = SygusV2Parser() if options.sygus_standard == 2 else SygusV1Parser()
        self.problem: Program = self.parser.parse(problem)
        self.symbol_table = SymbolTableBuilder.run(self.problem)
        self.printer = SygusV2ASTPrinter(self.symbol_table) if options.sygus_standard == 2 else SygusV1ASTPrinter(
            self.symbol_table, options.options)

        self.context = SynthesisProblemContext()
        self.context.enumerator_solver.set('smt.macro_finder', True)
        self.context.verification_solver.set('smt.macro_finder', True)
        self.context.smt_problem = self.convert_sygus_to_smt()
        self.context.constraints = [x for x in self.problem.commands if x.command_kind == CommandKind.CONSTRAINT]

        self.initialise_z3_variables()
        self.initialise_z3_synth_functions()
        self.initialise_z3_predefined_functions()
        self.parse_constraints()

    def print_msg(self, msg: str, level: int = 0) -> None:
        """
        Print a message based on the specified verbosity level.

        :param msg: The message to print.
        :param level: The verbosity level required to print the message (default: 0).
        """
        if self.options.verbose <= level:
            print(msg)

    def __str__(self) -> str:
        """
        Return the string representation of the synthesis problem.
        """
        return self.printer.run(self.problem, self.symbol_table)

    def info(self) -> None:
        """
        Print the string representation of the synthesis problem.
        """
        self.print_msg(str(self), level=1)

    def convert_sygus_to_smt(self) -> str:
        """
        Convert the synthesis problem from SyGuS format to SMT-LIB format.

        :return: The synthesis problem in SMT-LIB format.
        """
        i_expr = pyparsing.QuotedString(quoteChar='"') | pyparsing.QuotedString(quoteChar='|', unquoteResults=False)
        s_expr = pyparsing.nestedExpr(opener='(', closer=')', ignoreExpr=i_expr)
        s_expr.ignore(';' + pyparsing.restOfLine)

        sygus_parser = pyparsing.ZeroOrMore(s_expr)
        sygus_ast = sygus_parser.parseString(self.input_problem, parseAll=True).asList()

        constraints = []
        constraint_indices = []
        for i, statement in enumerate(sygus_ast):
            if statement[0] == 'constraint':
                constraints.append(statement[1])
                constraint_indices.append(i)
            elif statement[0] == 'check-synth':
                statement[0] = 'check-sat'
            elif statement[0] == 'synth-fun':
                statement[0] = 'declare-fun'
                statement[2] = [var_decl[1] for var_decl in statement[2]]
        if constraints:
            conjoined_constraints = ['and'] + constraints
            sygus_ast[constraint_indices[0]] = ['assert', conjoined_constraints]
            for index in reversed(constraint_indices[1:]):
                del sygus_ast[index]

        def serialise(line):
            return line if type(line) is not list else f'({" ".join(serialise(expression) for expression in line)})'

        return '\n'.join(serialise(statement) for statement in sygus_ast)

    def get_logic(self) -> str:
        """
        Get the logic of the synthesis problem.

        :return: The logic name.
        """
        return self.symbol_table.logic_name

    def get_synth_funcs(self) -> Dict[str, FunctionDescriptor]:
        """
        Get the synthesis functions of the problem.

        :return: A dictionary mapping function names to their declaration commands.
        """
        return self.symbol_table.synth_functions

    def get_predefined_funcs(self) -> Dict[str, FunctionDescriptor]:
        """
        Get the predefined functions of the problem.

        :return: A dictionary mapping function names to their declaration commands.
        """
        return self.symbol_table.user_defined_functions

    def get_synth_func(self, symbol: str) -> FunctionDescriptor:
        """
        Get a specific synthesis function by its symbol.

        :param symbol: The symbol of the synthesis function.
        :return: The function declaration command.
        """
        return next(filter(lambda x:
                           x.function_kind == FunctionKind.SYNTH_FUN and x.identifier.symbol == symbol,
                           list(self.symbol_table.synth_functions.values())))

    def get_var_symbols(self) -> List[str]:
        """
        Get the variable symbols of the problem.

        :return: A list of variable symbols.
        """
        return [x.symbol for x in self.problem.commands if x.command_kind == CommandKind.DECLARE_VAR]

    def get_function_symbols(self) -> List[str]:
        """
        Get the function symbols of the problem.

        :return: A list of function symbols.
        """
        return [x.symbol for x in self.problem.commands if x.command_kind == CommandKind.DECLARE_FUN]

    def initialise_z3_variables(self) -> None:
        """
        Initialize the Z3 variables.
        """
        for command in self.problem.commands:
            if command.command_kind == CommandKind.DECLARE_VAR and command.sort_expression.identifier.symbol == 'Int':
                self.context.z3_variables[command.symbol] = z3.Int(command.symbol)

    def initialise_z3_synth_functions(self) -> None:
        """
        Initialize the Z3 synthesis functions.
        """
        for func in self.get_synth_funcs().values():
            z3_arg_sorts = [self.convert_sort_descriptor_to_z3_sort(s) for s in func.argument_sorts]
            z3_range_sort = self.convert_sort_descriptor_to_z3_sort(func.range_sort)
            args = [z3.Int(name) for name in func.argument_names]
            # args = [z3.Const(f"arg_{i}", sort) for i, sort in enumerate(z3_arg_sorts)] # if z3.Int() doesn't work
            arg_mapping = dict(zip(func.argument_names, args))
            self.context.z3_synth_function_args[func.identifier.symbol] = arg_mapping
            self.context.z3_synth_functions[func.identifier.symbol] = z3.Function(func.identifier.symbol, *z3_arg_sorts,
                                                                                  z3_range_sort)

    def initialise_z3_predefined_functions(self) -> None:
        """
        Initialize the Z3 predefined functions.
        """
        for func in self.get_predefined_funcs().values():
            z3_arg_sorts = [self.convert_sort_descriptor_to_z3_sort(s) for s in func.argument_sorts]
            z3_range_sort = self.convert_sort_descriptor_to_z3_sort(func.range_sort)
            self.context.z3_predefined_functions[func.identifier.symbol] = z3.Function(func.identifier.symbol,
                                                                                       *z3_arg_sorts,
                                                                                       z3_range_sort)

    @staticmethod
    def negate_assertions(assertions: List[z3.ExprRef]) -> List[z3.ExprRef]:
        """
        Negate a list of assertions.

        :param assertions: The list of assertions to negate.
        :return: The negated assertions.
        """
        negated_assertions = []
        for assertion in assertions:
            args = assertion.num_args()
            if z3.is_and(assertion) or z3.is_or(assertion) or z3.is_not(assertion):
                if args > 1:
                    negated_children = [z3.Not(assertion.arg(i)) for i in range(args)]
                    negated_assertions.append(z3.Or(*negated_children))
                else:
                    negated_assertions.append(z3.Not(assertion))
            elif z3.is_expr(assertion) and args == 2:
                if z3.is_eq(assertion):
                    negated_assertions.append(assertion.arg(0) != assertion.arg(1))
                elif z3.is_ge(assertion):
                    negated_assertions.append(assertion.arg(0) < assertion.arg(1))
                elif z3.is_gt(assertion):
                    negated_assertions.append(assertion.arg(0) <= assertion.arg(1))
                elif z3.is_le(assertion):
                    negated_assertions.append(assertion.arg(0) > assertion.arg(1))
                elif z3.is_lt(assertion):
                    negated_assertions.append(assertion.arg(0) >= assertion.arg(1))
                else:
                    raise ValueError("Unsupported assertion type: {}".format(assertion))
        return negated_assertions

    def parse_constraints(self) -> None:
        """
        Parse the constraints of the synthesis problem.
        """
        all_constraints = []
        declared_variables = set(self.get_var_symbols())
        declared_functions = set(self.get_function_symbols())
        declared_synth_functions = set(self.get_synth_funcs().keys())

        for constraint in self.context.constraints:
            if isinstance(constraint, ast.ConstraintCommand):
                undeclared_variables = self.find_undeclared_variables(constraint.constraint, declared_variables,
                                                                      declared_functions, declared_synth_functions)
                if undeclared_variables:
                    raise ValueError(f"Undeclared variables used in constraint: {', '.join(undeclared_variables)}")
                term = self.parse_term(constraint.constraint)
                all_constraints.append(term)
                self.context.original_assertions.append(term)

        if all_constraints:
            combined_constraint = z3.And(*all_constraints)
            self.context.z3_constraints.append(combined_constraint)
        else:
            self.print_msg("Warning: No constraints found or generated.", level=1)

        self.context.negated_constraints = self.negate_assertions(self.context.z3_constraints)
        self.print_msg(f"Negated constraints: {self.context.negated_constraints}.", level=1)

    def find_undeclared_variables(self, term, declared_variables, declared_functions, declared_synth_functions):
        """
        Find undeclared variables in a term.

        :param term: The term to check.
        :param declared_variables: The set of declared variables.
        :param declared_functions: The set of declared functions.
        :param declared_synth_functions: The set of declared synthesis functions.
        :return: A list of undeclared variables found in the term.
        """
        undeclared_variables = []

        if isinstance(term, ast.IdentifierTerm):
            symbol = term.identifier.symbol
            if symbol not in declared_variables and symbol not in declared_functions and symbol not in declared_synth_functions:
                undeclared_variables.append(symbol)
        elif isinstance(term, ast.FunctionApplicationTerm):
            for arg in term.arguments:
                undeclared_variables.extend(self.find_undeclared_variables(arg, declared_variables, declared_functions,
                                                                           declared_synth_functions))
        elif isinstance(term, ast.QuantifiedTerm):
            for var_name, _ in term.quantified_variables:
                if var_name not in declared_variables:
                    undeclared_variables.append(var_name)
            undeclared_variables.extend(
                self.find_undeclared_variables(term.term_body, declared_variables, declared_functions,
                                               declared_synth_functions))

        return undeclared_variables

    def parse_term(self, term: ast.Term) -> ExprRef | FuncDeclRef | bool | int:
        """
        Parse a term from the AST and convert it to a Z3 expression.

        :param term: The term to parse.
        :return: The Z3 expression representing the term.
        """
        if isinstance(term, ast.IdentifierTerm):
            symbol = term.identifier.symbol
            if symbol in self.context.z3_variables:
                return self.context.z3_variables[symbol]
            elif symbol in self.context.z3_synth_functions:
                return self.context.z3_synth_functions[symbol]
            elif symbol in self.context.z3_predefined_functions:
                return self.context.z3_predefined_functions[symbol]
            else:
                raise ValueError(f"Undefined symbol: {symbol}")
        elif isinstance(term, ast.LiteralTerm):
            literal = term.literal
            if literal.literal_kind == ast.LiteralKind.NUMERAL:
                return z3.IntVal(int(literal.literal_value))
            elif literal.literal_kind == ast.LiteralKind.BOOLEAN:
                return z3.BoolVal(literal.literal_value.lower() == "true")
            else:
                raise ValueError(f"Unsupported literal kind: {literal.literal_kind}")
        elif isinstance(term, ast.FunctionApplicationTerm):
            func_symbol = term.function_identifier.symbol
            args = [self.parse_term(arg) for arg in term.arguments]
            operator_map = {
                "and": lambda *args: z3.And(*args),
                "or": lambda *args: z3.Or(*args),
                "not": lambda arg: z3.Not(arg),
                ">": lambda arg1, arg2: arg1 > arg2,
                "<": lambda arg1, arg2: arg1 < arg2,
                ">=": lambda arg1, arg2: arg1 >= arg2,
                "<=": lambda arg1, arg2: arg1 <= arg2,
                "+": lambda *args: z3.Sum(args),
                "*": lambda *args: z3.Product(*args),
                "/": lambda arg1, arg2: arg1 / arg2,
                "=>": lambda arg1, arg2: z3.Implies(arg1, arg2),
            }
            if func_symbol in operator_map:
                op = operator_map[func_symbol]
                if func_symbol == "not":
                    assert len(args) == 1, "'not' should have 1 argument"
                    return op(args[0])
                else:
                    assert len(args) >= 2, f"'{func_symbol}' should have at least 2 arguments"
                    return op(*args)
            elif func_symbol == "-":
                if len(args) == 1:
                    return -args[0]
                elif len(args) == 2:
                    return args[0] - args[1]
                raise ValueError("Minus operator '-' should have 1 or 2 arguments")
            elif func_symbol == "=":
                if len(args) == 2:
                    return args[0] == args[1]
                else:
                    return z3.And(*[args[i] == args[i + 1] for i in range(len(args) - 1)])
            elif func_symbol in self.context.z3_synth_functions:
                func_term = self.context.z3_synth_functions[func_symbol]
                return func_term(*args)
            elif func_symbol in self.context.z3_predefined_functions:
                func = self.context.z3_predefined_functions[func_symbol]
                return func(*args)
            else:
                raise ValueError(f"Undefined function symbol: {func_symbol}")
        elif isinstance(term, ast.QuantifiedTerm):
            quantified_variables = []
            for var_name, _ in term.quantified_variables:
                if var_name in self.context.z3_variables:
                    quantified_variables.append(self.context.z3_variables[var_name])
                else:
                    raise ValueError(f"Undeclared variable used in quantifier: {var_name}")
            body = self.parse_term(term.term_body)
            if term.quantifier_kind == ast.QuantifierKind.FORALL:
                return z3.ForAll(quantified_variables, body)
            elif term.quantifier_kind == ast.QuantifierKind.EXISTS:
                return z3.Exists(quantified_variables, body)
            else:
                raise ValueError(f"Unsupported quantifier kind: {term.quantifier_kind}")
        else:
            raise ValueError(f"Unsupported term type: {type(term)}")

    @staticmethod
    def convert_sort_descriptor_to_z3_sort(sort_descriptor: SortDescriptor) -> z3.SortRef | None:
        """
        Convert a sort descriptor to a Z3 sort.

        :param sort_descriptor: The sort descriptor to convert.
        :return: The corresponding Z3 sort.
        """
        sort_symbol = sort_descriptor.identifier.symbol
        return {
            'Int': z3.IntSort(),
            'Bool': z3.BoolSort(),
        }.get(sort_symbol, None)

    def generate_correct_abs_max_function(self, arg_sorts: List[z3.SortRef]) -> Tuple[Callable, str]:
        """
        Generate a correct implementation of the absolute_max function.

        :param arg_sorts: The arguments of the function.
        :return: A tuple containing the function implementation and its string representation.
        """
        args = [z3.Var(i, sort) for i, sort in enumerate(arg_sorts)]

        def absolute_max_function(*values):
            if len(values) != 2:
                raise ValueError("absolute_max_function expects exactly 2 arguments.")
            x, y = values
            return If(If(x >= 0, x, -x) > If(y >= 0, y, -y), If(x >= 0, x, -x), If(y >= 0, y, -y))

        expr = absolute_max_function(*args[:2])
        func_str = f"def absolute_max_function({', '.join(str(arg) for arg in args[:2])}):\n"
        func_str += f"    return {str(expr)}\n"
        return absolute_max_function, func_str

    def generate_max_function(self, arg_sorts: List[z3.SortRef]) -> Tuple[Callable, str]:
        """
        Generate a correct implementation of the max function.

        :param arg_sorts: The arguments of the function.
        :return: A tuple containing the function implementation and its string representation.
        """
        args = [z3.Var(i, sort) for i, sort in enumerate(arg_sorts)]

        def max_function(*values):
            if len(values) != 2:
                raise ValueError("max_function expects exactly 2 arguments.")
            x, y = values
            return If(x <= y, y, x)

        expr = max_function(*args[:2])
        func_str = f"def max_function({', '.join(str(arg) for arg in args[:2])}):\n"
        func_str += f"    return {str(expr)}\n"
        return max_function, func_str

    def generate_valid_other_solution_one(self, arg_sorts: List[z3.SortRef]) -> Tuple[Callable, str]:
        """
        Generate a solution that breaks the commutativity constraint.

        :param arg_sorts: The arguments of the function.
        :return: A tuple containing the function implementation and its string representation.
        """
        args = [z3.Var(i, sort) for i, sort in enumerate(arg_sorts)]

        def invalid_function(*values):
            if len(values) != 2:
                raise ValueError("invalid_function expects exactly 2 arguments.")
            x, y = values
            return If(x > y, If(x > y, x, 1), y - 0)

        expr = invalid_function(*args[:2])
        func_str = f"def invalid_function_one({', '.join(str(arg) for arg in args[:2])}):\n"
        func_str += f"    return {str(expr)}\n"
        return invalid_function, func_str

    def generate_invalid_solution_two(self, arg_sorts: List[z3.SortRef]) -> Tuple[Callable, str]:
        """
        Generate a solution that breaks the commutativity constraint.

        :param arg_sorts: A list of the sorts of the parameters for the function
        :return: A tuple containing the function implementation and its string representation.
        """
        args = [z3.Var(i, sort) for i, sort in enumerate(arg_sorts)]

        def invalid_function(*values):
            if len(values) != 2:
                raise ValueError("invalid_function expects exactly 2 arguments.")
            x, y = values
            return If(x > y, x, y - 1)

        expr = invalid_function(*args[:2])
        func_str = f"def invalid_function_two({', '.join(str(arg) for arg in args[:2])}):\n"
        func_str += f"    return {str(expr)}\n"
        return invalid_function, func_str

    def generate_arithmetic_function(self, arg_sorts: List[z3.SortRef], depth: int, complexity: int,
                                     operations: List[str] = None) -> Tuple[Callable, str]:
        """
        Generate an arithmetic function based on the given number of arguments, argument sorts, depth, complexity, and operations.
    
        :param arg_sorts: The list of sorts for each argument.
        :param depth: The maximum depth of the generated expression.
        :param complexity: The maximum complexity of the generated expression.
        :param operations: The list of allowed operations (default: ['+', '-', '*', 'If', 'Neg']).
        :return: A tuple containing the function implementation and its string representation.
        """
        if operations is None:
            operations = ['+', '-', '*', 'If', 'Neg']

        args = [z3.Var(i, sort) for i, sort in enumerate(arg_sorts)]
        num_args = len(args)

        def generate_expression(curr_depth: int, curr_complexity: int) -> z3.ExprRef:
            if curr_depth == 0 or curr_complexity == 0:
                return random.choice(args) if args and random.random() < 0.5 else z3.IntVal(
                    random.randint(self.MIN_CONST, self.MAX_CONST))

            op = random.choice(operations)
            if op == 'If' and num_args >= 2:
                condition = random.choice(
                    [args[i] < args[j] for i in range(num_args) for j in range(i + 1, num_args)] +
                    [args[i] <= args[j] for i in range(num_args) for j in range(i + 1, num_args)] +
                    [args[i] > args[j] for i in range(num_args) for j in range(i + 1, num_args)] +
                    [args[i] >= args[j] for i in range(num_args) for j in range(i + 1, num_args)] +
                    [args[i] == args[j] for i in range(num_args) for j in range(i + 1, num_args)] +
                    [args[i] != args[j] for i in range(num_args) for j in range(i + 1, num_args)]
                )
                true_expr = generate_expression(curr_depth - 1, curr_complexity - 1)
                false_expr = generate_expression(curr_depth - 1, curr_complexity - 1)
                return z3.If(condition, true_expr, false_expr)
            elif op == 'If' and num_args == 1:
                condition = random.choice(
                    [args[0] < z3.IntVal(random.randint(self.MIN_CONST, self.MAX_CONST)),
                     args[0] <= z3.IntVal(random.randint(self.MIN_CONST, self.MAX_CONST)),
                     args[0] > z3.IntVal(random.randint(self.MIN_CONST, self.MAX_CONST)),
                     args[0] >= z3.IntVal(random.randint(self.MIN_CONST, self.MAX_CONST)),
                     args[0] == z3.IntVal(random.randint(self.MIN_CONST, self.MAX_CONST)),
                     args[0] != z3.IntVal(random.randint(self.MIN_CONST, self.MAX_CONST))]
                )
                true_expr = generate_expression(curr_depth - 1, curr_complexity - 1)
                false_expr = generate_expression(curr_depth - 1, curr_complexity - 1)
                return z3.If(condition, true_expr, false_expr)
            elif op == 'Neg':
                expr = generate_expression(curr_depth - 1, curr_complexity - 1)
                return -expr
            elif op in ['+', '-', '*']:
                left_expr = generate_expression(curr_depth - 1, curr_complexity - 1)
                right_expr = generate_expression(curr_depth - 1, curr_complexity - 1)
                if op == '+':
                    return left_expr + right_expr
                elif op == '-':
                    return left_expr - right_expr
                elif op == '*':
                    return left_expr * right_expr
            else:
                raise ValueError(f"Unsupported operation: {op}")

        expr = generate_expression(depth, complexity)
        self.print_msg(f"Generated expression: {expr}", level=1)
        self.print_msg(f"Expression type: {type(expr)}", level=1)

        def arithmetic_function(*values):
            if len(values) != num_args:
                raise ValueError("Incorrect number of values provided.")
            simplified_expr = z3.simplify(z3.substitute(expr, [(arg, value) for arg, value in zip(args, values)]))
            return simplified_expr

        func_str = f"def arithmetic_function({', '.join(f'arg{i}' for i in range(num_args))}):\n"
        func_str += f"    return {str(expr)}\n"

        return arithmetic_function, func_str

    def substitute_constraints(self, constraints: Collection[z3.ExprRef], func: z3.FuncDeclRef,
                               candidate_function: typing.Union[z3.FuncDeclRef, z3.QuantifierRef, Callable]) -> List[
        ExprRef]:
        """
        Substitute a candidate expression into a list of constraints.

        :param constraints: The list of constraints.
        :param func: The function to substitute.
        :param candidate_function: The candidate function to substitute.
        :return: The substituted constraints.
        """
        substituted_constraints = [substitute_funs(constraint, (func, candidate_function)) for constraint in
                                   constraints]
        self.print_msg(f"substituted_constraints {substituted_constraints}", level=0)
        return substituted_constraints

    def test_candidate(self, constraints: List[z3.ExprRef], negated_constraints: Collection[z3.ExprRef], func_str: str,
                       func: z3.FuncDeclRef, args: List[z3.ExprRef], candidate_function: typing.Union[
                z3.FuncDeclRef, z3.QuantifierRef, Callable, z3.ExprRef]) -> bool:
        """
        Test a candidate expression against the constraints and negated constraints.
    
        :param constraints: The list of constraints.
        :param negated_constraints: The list of negated constraints.
        :param func_str: The string representation of the function.
        :param func: The function to test.
        :param args: The arguments of the function.
        :param candidate_function: The candidate expression to test.
        :return: True if the candidate expression satisfies the constraints, False otherwise.
        """

        self.context.verification_solver.reset()
        substituted_constraints = self.substitute_constraints(constraints, func, candidate_function)
        self.context.verification_solver.add(substituted_constraints)

        if self.context.verification_solver.check() == unsat:
            self.print_msg(f"Verification failed for guess {func_str}. Candidate violates constraints.", level=0)
            return False

        self.context.enumerator_solver.reset()
        substituted_neg_constraints = self.substitute_constraints(negated_constraints, func, candidate_function)
        self.context.enumerator_solver.add(substituted_neg_constraints)

        if self.context.enumerator_solver.check() == sat:
            model = self.context.enumerator_solver.model()
            counterexample: Dict[str, ExprRef] = {var.name(): model.get_interp(model.decls()[i]) for i, var in
                                                  enumerate(model.decls())}
            incorrect_output = None
            if callable(candidate_function):
                incorrect_output = model.eval(candidate_function(*args), model_completion=True)
            elif isinstance(candidate_function, (QuantifierRef, ExprRef)):
                incorrect_output = model.eval(candidate_function, model_completion=True)

            self.context.counterexamples.append((counterexample, incorrect_output))
            self.print_msg(f"Incorrect output for {func_str}: {counterexample} == {incorrect_output}", level=0)
            return False
        else:
            self.print_msg(f"No counterexample found! Guess should be correct: {func_str}.", level=0)
            return True
        
    def substitute_constraints_multiple(self, constraints: Collection[z3.ExprRef], funcs: List[z3.FuncDeclRef],
                                        candidate_functions: List[typing.Union[z3.FuncDeclRef, z3.QuantifierRef, z3,ExprRef,  Callable]]) -> List[
        z3.ExprRef]:
        """
        Substitute candidate expressions into a list of constraints.
    
        :param constraints: The list of constraints.
        :param funcs: The list of functions to substitute.
        :param candidate_functions: The list of candidate functions to substitute.
        :return: The substituted constraints.
        """
        substitutions = list(zip(funcs, candidate_functions))
        substituted_constraints = [substitute_funs(constraint, substitutions) for constraint in constraints]
        self.print_msg(f"substituted_constraints {substituted_constraints}", level=0)
        return substituted_constraints

    def test_multiple_candidates(self, constraints: List[z3.ExprRef], negated_constraints: Collection[z3.ExprRef],
                                 func_strs: List[str], candidate_functions: List[z3.ExprRef],
                                 args_list: List[List[z3.SortRef]]) -> bool:
        """
        Test multiple candidate functions.

        :param constraints: The list of constraints.
        :param negated_constraints: The list of negated constraints.
        :param func_strs: The string representations of the functions.
        :param candidate_functions: The candidate expressions to test.
        :param args_list: The arguments of the functions.
        :return: True if the candidate expressions satisfy the constraints, False otherwise.
        """

        self.context.verification_solver.reset()
        substituted_constraints = self.substitute_constraints_multiple(constraints,list(self.context.z3_synth_functions.values()),candidate_functions)
        self.context.verification_solver.add(substituted_constraints)

        if self.context.verification_solver.check() == unsat:
            self.print_msg(f"Verification failed for guess {'; '.join(func_strs)}. Candidates violate constraints.",
                           level=0)
            return False

        self.context.enumerator_solver.reset()
        substituted_neg_constraints = self.substitute_constraints_multiple(negated_constraints,list(self.context.z3_synth_functions.values()),candidate_functions)
        self.context.enumerator_solver.add(substituted_neg_constraints)

        if self.context.enumerator_solver.check() == sat:
            model = self.context.enumerator_solver.model()
            counterexamples = []
            incorrect_outputs = []

            for func, candidate, args in zip(func_strs, candidate_functions, args_list):
                free_variables = [z3.Var(i, sort) for i, sort in enumerate(args)]
                incorrect_output = None
                if callable(candidate):
                    incorrect_output = model.eval(candidate(*free_variables), model_completion=True)
                elif isinstance(candidate, (z3.QuantifierRef, z3.ExprRef)):
                    incorrect_output = model.eval(candidate, model_completion=True)

                counterexample: Dict[str, ExprRef] = {var.name(): model.get_interp(model.decls()[i]) for i, var in enumerate(model.decls())}
                counterexamples.append(counterexample)
                incorrect_outputs.append(incorrect_output)
                self.context.counterexamples.append((counterexample, incorrect_output))

            self.print_msg(f"Incorrect outputs for {'; '.join(func_strs)}: {incorrect_outputs}", level=0)
            return False
        else:
            self.print_msg(f"No counterexample found! Guesses should be correct: {'; '.join(func_strs)}.", level=0)
            return True

    def execute_cegis(self) -> None:
        """
        Execute the chosen counterexample-guided inductive synthesis algorithm.
        """
        max_complexity = 3
        max_depth = 3
        max_candidates_to_evaluate_at_each_depth = 10
        args_list = []

        for func in self.context.z3_synth_functions.values():
            args = [func.domain(i) for i in range(func.arity())]
            args_list.append(args)

        tested_candidates = set()

        for depth in range(1, max_depth + 1):
            for complexity in range(1, max_complexity + 1):
                guesses = []
                for _ in range(max_candidates_to_evaluate_at_each_depth):
                    candidates = []
                    func_strs = []
                    for args in args_list:
                        candidate, func_str = self.generate_arithmetic_function(args, depth, complexity)
                        candidates.append(candidate)
                        func_strs.append(func_str)

                    free_variables_list = [[z3.Var(i, sort) for i, sort in enumerate(args)] for args in args_list]
                    simplified_candidates = [z3.simplify(candidate(*free_variables)) for candidate, free_variables in
                                             zip(candidates, free_variables_list)]

                    if str(simplified_candidates) not in tested_candidates:
                        tested_candidates.add(str(simplified_candidates))
                        guesses.append((candidates, func_strs))

                for candidates, func_strs in guesses:
                    try:
                        candidate_functions = []
                        for candidate, args in zip(candidates, args_list):
                            free_variables = [z3.Var(i, sort) for i, sort in enumerate(args)]
                            candidate_function = candidate(*free_variables)
                            candidate_functions.append(candidate_function)

                        self.print_msg(f"candidate_functions for substitution {candidate_functions}", level=0)
                        self.print_msg(
                            f"Testing guess (complexity: {complexity}, depth: {depth}): {'; '.join(func_strs)}",
                            level=1)
                        result = self.test_multiple_candidates(self.context.z3_constraints,
                                                               self.context.negated_constraints, func_strs,
                                                               candidate_functions, args_list)
                        self.print_msg("\n", level=1)
                        if result:
                            self.print_msg(f"Found satisfying candidates! {'; '.join(func_strs)}", level=0)
                            self.print_msg("-" * 150, level=0)
                            return
                        self.print_msg("-" * 75, level=0)
                    except Exception as e:
                        self.print_msg(f"Error occurred while testing candidates: {'; '.join(func_strs)}", level=0)
                        self.print_msg(f"Error message: {str(e)}", level=0)
                        self.print_msg("Skipping these candidates.", level=0)
                        self.print_msg("\n", level=1)
                        raise

        self.print_msg("No satisfying candidates found.", level=0)
        # following is just for verifying, will convert these to unit tests
        if sum(len(sublist) for sublist in args_list) == 4:
            self.print_msg("Trying known candidate for 4.", level=0)

            def id1_function(*values):
                x = values[0]
                return x

            def id2_function(*values):
                x = values[0]
                return x

            def id3_function(*values):
                x = values[0]
                return x

            def id4_function(*values):
                x = values[0]
                return x

            candidate_functions = [id1_function, id2_function, id3_function, id4_function]
            func_strs = ["id1_function", "id2_function", "id3_function", "id4_function"]
            args_list = [z3.Var(0, IntSort())]
            candidate_functions = [ f(*args_list) for f in candidate_functions ]
            self.print_msg(f"candidate_functions for substitution {candidate_functions}", level=0)
            self.print_msg(f"Testing known candidate: {'; '.join(func_strs)}", level=1)
            result = self.test_multiple_candidates(self.context.z3_constraints, self.context.negated_constraints, func_strs,
                                                   candidate_functions, args_list)
            self.print_msg("\n", level=1)
            if result:
                self.print_msg(f"Found a satisfying candidate! {'; '.join(func_strs)}", level=0)
                self.print_msg("-" * 150, level=0)
                return
            self.print_msg("-" * 75, level=0)

        if sum(len(sublist) for sublist in args_list) == 2:
            self.print_msg("Trying known candidate for max", level=0)
            free_variables = [Var(i, func.domain(i)) for i in range(list(self.context.z3_synth_functions.values())[0].arity())]
            args = [func.domain(i) for i in range(list(self.context.z3_synth_functions.values())[0].arity())]
            candidate, func_str = self.generate_max_function(args)
            candidate_function = candidate(*free_variables)
            self.print_msg(f"candidate_function for substitution {candidate_function}", level=0)
            self.print_msg(f"Testing guess: {func_str}", level=1)
            result = self.test_multiple_candidates(self.context.z3_constraints, self.context.negated_constraints, [func_str],
                                                   [candidate_function], args )
            self.print_msg("\n", level=1)
            if result:
                self.print_msg(f"Found a satisfying candidate! {func_str}", level=0)
                self.print_msg("-" * 150, level=0)
                return
            
        if sum(len(sublist) for sublist in args_list) == 3:
            self.print_msg("Trying known candidate for max3", level=0)
            free_variables = [Var(i, func.domain(i)) for i in range(list(self.context.z3_synth_functions.values())[0].arity())]
            args = [func.domain(i) for i in range(list(self.context.z3_synth_functions.values())[0].arity())]
            def max_function_3(*values):
                x, y , z = values
                return z3.If(x >= y, z3.If(x >= z, x, z), z3.If(y >= z, y, z))
            
            func_str = "max_3_ints"
            candidate_function = max_function_3(*free_variables)
            self.print_msg(f"candidate_function for substitution {candidate_function}", level=0)
            self.print_msg(f"Testing guess: {func_str}", level=1)
            result = self.test_multiple_candidates(self.context.z3_constraints, self.context.negated_constraints, [func_str],
                                                   [candidate_function], args )
            self.print_msg("\n", level=1)
            if result:
                self.print_msg(f"Found a satisfying candidate! {func_str}", level=0)
                self.print_msg("-" * 150, level=0)
                return