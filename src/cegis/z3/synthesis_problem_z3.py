import random
import typing
from typing import List, Dict, Tuple, Set, Callable, Collection, Any

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
    counterexamples: List[
        Tuple[QuantifierRef | ExprRef | Callable | Any, Dict[str, ExprRef], ExprRef]] = dataclasses.field(
        default_factory=list)
    z3_negated_constraints: Set[ExprRef] = dataclasses.field(default_factory=set)
    additional_constraints: List[ExprRef] = dataclasses.field(default_factory=list)
    synth_functions: List[Dict[str, typing.Union[str, List[ExprRef], SortRef]]] = dataclasses.field(
        default_factory=list)
    smt_problem: str = ""
    variable_mapping_dict: Dict[str, Dict[z3.ExprRef, z3.ExprRef]] = dataclasses.field(default_factory=dict)
    all_z3_functions: Dict[str, z3.FuncDeclRef] = dataclasses.field(default=dict)


class SynthesisProblem:
    """
    A class representing a synthesis problem in the SyGuS format.
    """
    MIN_CONST = -10
    MAX_CONST = 10
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
        self.map_z3_variables()
        self.populate_all_z3_functions()
        self.parse_constraints()

    def print_msg(self, msg: str, level: int = 0) -> None:
        """
        Print a message based on the specified verbosity level.

        :param msg: The message to print.
        :param level: The verbosity level required to print the message (default: 0).
        """
        if self.options.verbose <= level:
            print(msg)

    def __str__(self, as_smt=False) -> str:
        """
        Return the string representation of the synthesis problem.
        """
        if as_smt:
            return self.context.smt_problem
        return self.printer.run(self.problem, self.symbol_table)

    def info_sygus(self) -> None:
        """
        Print the string representation of the synthesis problem.
        """
        self.print_msg(str(self), level=1)

    def info_smt(self) -> None:
        """
        Print the string representation of the synthesis problem.
        """
        self.print_msg(self.__str__(as_smt=True), level=1)

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
    def populate_all_z3_functions(self) -> None:
        """
        Add all parsed z3 functions to a dictionary.
        """
        self.context.all_z3_functions = dict(self.context.z3_synth_functions.items())
        self.context.all_z3_functions.update(self.context.z3_predefined_functions.items())
    
    def map_z3_variables(self) -> None:
        """
        Map z3 variables.
        """
        for func_name, func in self.context.z3_synth_functions.items():
            free_variables = [z3.Var(i, func.domain(i)) for i in range(func.arity())]
            declared_variables = list(self.context.z3_variables.values())
            variable_mapping = {free_var: declared_var for free_var, declared_var in
                                zip(free_variables, declared_variables)}
            self.context.variable_mapping_dict[func_name] = variable_mapping

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

        self.context.z3_negated_constraints = self.negate_assertions(self.context.z3_constraints)
        self.print_msg(f"Negated constraints: {self.context.z3_negated_constraints}.", level=1)

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
                "ite": lambda cond, arg1, arg2: z3.If(cond, arg1, arg2),
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
        elif isinstance(term, ast.DefineFunCommand):
            func_name = term.function_name
            func_args = [z3.Var(arg[0], self.convert_sort_descriptor_to_z3_sort(arg[1])) for arg in
                         term.function_parameters]
            func_body = self.parse_term(term.function_body)
            func = z3.Function(func_name, *[arg.sort() for arg in func_args], func_body.sort())
            self.context.z3_predefined_functions[func_name] = func
            return func
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

            expr = None
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
                expr = z3.If(condition, true_expr, false_expr)
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
                expr = z3.If(condition, true_expr, false_expr)
            elif op == 'Neg':
                expr = generate_expression(curr_depth - 1, curr_complexity - 1)
                if z3.is_bool(expr):
                    return z3.Not(expr)
                else:
                    return expr * -1
            elif op in ['+', '-', '*']:
                left_expr = generate_expression(curr_depth - 1, curr_complexity - 1)
                right_expr = generate_expression(curr_depth - 1, curr_complexity - 1)
                if op == '+':
                    expr = left_expr + right_expr
                elif op == '-':
                    expr = left_expr - right_expr
                elif op == '*':
                    expr = left_expr * right_expr
            else:
                raise ValueError(f"Unsupported operation: {op}")

            return expr

        generated_expression = generate_expression(depth, complexity)
        self.print_msg(f"Generated expression: {generated_expression}", level=1)
        self.print_msg(f"Expression type: {type(generated_expression)}", level=1)

        def arithmetic_function(*values):
            if len(values) != num_args:
                raise ValueError("Incorrect number of values provided.")
            simplified_expr = z3.simplify(
                z3.substitute(generated_expression, [(arg, value) for arg, value in zip(args, values)]))
            return simplified_expr

        func_str = f"def arithmetic_function({', '.join(f'arg{i}' for i in range(num_args))}):\n"
        func_str += f"    return {str(generated_expression)}\n"

        return arithmetic_function, func_str

    def substitute_constraints_multiple(self, constraints: Collection[z3.ExprRef],
                                        functions_to_replace: List[z3.FuncDeclRef],
                                        candidate_functions: List[
                                            typing.Union[z3.FuncDeclRef, z3.QuantifierRef, z3, ExprRef, Callable]]) -> \
            List[z3.ExprRef]:
        """
        Substitute candidate expressions into a list of constraints.
    
        :param constraints: The list of constraints.
        :param functions_to_replace: The list of functions to substitute.
        :param candidate_functions: The list of candidate functions to substitute.
        :return: The substituted constraints.
        """
        substitutions = list(zip(functions_to_replace, candidate_functions))
        substituted_constraints = [substitute_funs(constraint, substitutions) for constraint in constraints]
        self.print_msg(f"substituted_constraints {substituted_constraints}", level=0)
        return substituted_constraints

    def test_multiple_candidates(self, func_strs: List[str], candidate_functions: List[z3.ExprRef]) -> bool:
        """
        Test multiple candidate functions.

        :param func_strs: The string representations of the functions.
        :param candidate_functions: The candidate expressions to test.
        :return: True if the candidate expressions satisfy the constraints, False otherwise.
        """

        self.context.enumerator_solver.reset()
        substituted_neg_constraints = self.substitute_constraints_multiple(self.context.z3_negated_constraints, list(self.context.z3_synth_functions.values()), candidate_functions)
        self.context.enumerator_solver.add(substituted_neg_constraints)

        if self.context.enumerator_solver.check() == sat:
            model = self.context.enumerator_solver.model()
            counterexamples = []
            incorrect_outputs = []
            candidate_function_exprs = []

            for func, candidate, variable_mapping in zip(func_strs, candidate_functions,
                                                         self.context.variable_mapping_dict.values()):
                free_variables = list(variable_mapping.keys())
                counterexample = {str(free_var): model.eval(declared_var, model_completion=True).as_long()
                                  for free_var, declared_var in variable_mapping.items()}

                incorrect_output = z3.simplify(z3.substitute(candidate, [(arg, z3.IntVal(value)) for arg, value in
                                                                         zip(free_variables,
                                                                             list(counterexample.values()))]))

                self.print_msg(f"Counterexample: {counterexample}", level=0)
                counterexamples.append(counterexample)
                incorrect_outputs.append(incorrect_output)
                candidate_function_expr = candidate(*free_variables) if callable(candidate) else candidate
                candidate_function_exprs.append(candidate_function_expr)

                self.context.counterexamples.append((func, counterexample, incorrect_output))

            self.print_msg(f"Incorrect outputs for {'; '.join(func_strs)}: {incorrect_outputs}", level=0)
            return False
        else:
            self.context.verification_solver.reset()
            substituted_constraints = self.substitute_constraints_multiple(self.context.z3_constraints, list(
                self.context.z3_synth_functions.values()), candidate_functions)
            self.context.verification_solver.add(substituted_constraints)
            if self.context.verification_solver.check() == unsat:
                self.print_msg(f"Verification failed for guess {'; '.join(func_strs)}. Candidates violate constraints.",
                               level=0)
                return False
            self.print_msg(f"No counterexample found! Guesses should be correct: {'; '.join(func_strs)}.", level=0)
            return True

    def execute_cegis(self) -> None:
        """
        Execute the chosen counterexample-guided inductive synthesis algorithm.
        """
        max_complexity = 3
        max_depth = 3
        max_candidates_to_evaluate_at_each_depth = 10

        tested_candidates = set()

        for depth in range(1, max_depth + 1):
            for complexity in range(1, max_complexity + 1):
                guesses = []
                for _ in range(max_candidates_to_evaluate_at_each_depth):
                    candidates = []
                    func_strs = []
                    for func_name, variable_mapping in self.context.variable_mapping_dict.items():
                        candidate, func_str = self.generate_arithmetic_function(
                            [x.sort() for x in list(variable_mapping.keys())], depth, complexity)
                        candidates.append(candidate)
                        func_strs.append(func_str)

                    simplified_candidates = [z3.simplify(candidate(*free_variables)) for candidate, free_variables in
                                             zip(candidates, self.context.variable_mapping_dict.values())]

                    if str(simplified_candidates) not in tested_candidates:
                        tested_candidates.add(str(simplified_candidates))
                        guesses.append((candidates, func_strs, self.context.variable_mapping_dict))

                for candidates, func_strs, variable_mapping_dict in guesses:
                    try:
                        candidate_expressions = []
                        for candidate, variable_mapping in zip(candidates, variable_mapping_dict.values()):
                            free_variables = list(variable_mapping.keys())
                            candidate_expr_representation = candidate(*free_variables)
                            candidate_expressions.append(candidate_expr_representation)

                        self.print_msg(f"candidate_functions for substitution {candidate_expressions}", level=0)
                        self.print_msg(
                            f"Testing guess (complexity: {complexity}, depth: {depth}): {'; '.join(func_strs)}",
                            level=1)
                        result = self.test_multiple_candidates(func_strs,candidate_expressions)
                        self.print_msg("\n", level=1)
                        if result:
                            self.print_msg(f"Found satisfying candidates! {'; '.join(func_strs)}", level=0)
                            self.print_msg("-" * 150, level=0)
                            for func, counterexample, incorrect_output in self.context.counterexamples:
                                self.print_msg(
                                    f"Candidate function: {func} Args:{counterexample} Output: {incorrect_output}",
                                    level=0)

                            self.print_msg(f"Tested candidates: {tested_candidates}", level=0)
                            return
                        self.print_msg("-" * 75, level=0)
                    except Exception as e:
                        self.print_msg(f"Error occurred while testing candidates: {'; '.join(func_strs)}", level=0)
                        self.print_msg(f"Error message: {str(e)}", level=0)
                        self.print_msg("Skipping these candidates.", level=0)
                        self.print_msg("\n", level=1)
                        raise
        for func, counterexample, incorrect_output in self.context.counterexamples:
            self.print_msg(f"Candidate function: {func} Args:{counterexample} Output: {incorrect_output}", level=0)

        self.print_msg(f"Tested candidates: {tested_candidates}", level=0)
        self.print_msg("No satisfying candidates found.", level=0)
