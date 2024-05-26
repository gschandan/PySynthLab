import typing
from typing import List, Dict, Tuple, Union, Set,  Callable, Collection
from z3 import *
import pyparsing
import random

from z3 import ExprRef, FuncDeclRef, IntNumRef, BoolRef, Probe, ArithRef, PatternRef, QuantifierRef, RatNumRef, \
    AlgebraicNumRef, BitVecNumRef, BitVecRef, ArrayRef, DatatypeRef, FPNumRef, FPRef, FiniteDomainNumRef, \
    FiniteDomainRef, FPRMRef, SeqRef, CharRef, ReRef

from pysynthlab.helpers.parser.src import ast
from pysynthlab.helpers.parser.src.ast import Program, CommandKind
from pysynthlab.helpers.parser.src.resolution import FunctionKind, SortDescriptor, FunctionDescriptor
from pysynthlab.helpers.parser.src.symbol_table_builder import SymbolTableBuilder
from pysynthlab.helpers.parser.src.v1.parser import SygusV1Parser
from pysynthlab.helpers.parser.src.v1.printer import SygusV1ASTPrinter
from pysynthlab.helpers.parser.src.v2.parser import SygusV2Parser
from pysynthlab.helpers.parser.src.v2.printer import SygusV2ASTPrinter

class SynthesisProblem:
    """
    A class representing a synthesis problem in the SyGuS format.
    """
    MIN_CONST = -2
    MAX_CONST = 2
    pyparsing.ParserElement.enablePackrat()

    def __init__(self, problem: str, sygus_standard: int = 1, options: object = None):
        """
        Initialize a SynthesisProblem instance.

        :param problem: The input problem in the SyGuS format.
        :param sygus_standard: The SyGuS standard version (default: 1).
        :param options: Additional options (default: None).
        """
        if options is None:
            options = {}

        self.input_problem: str = problem
        self.options = options
        self.sygus_standard = sygus_standard
        self.parser = SygusV2Parser() if sygus_standard == 2 else SygusV1Parser()
        self.problem: Program = self.parser.parse(problem)
        self.symbol_table = SymbolTableBuilder.run(self.problem)
        self.printer = SygusV2ASTPrinter(self.symbol_table) if sygus_standard == 2 else SygusV1ASTPrinter(self.symbol_table, options)

        self.enumerator_solver = z3.Solver()
        self.enumerator_solver.set('smt.macro_finder', True)

        self.verification_solver = z3.Solver()
        self.verification_solver.set('smt.macro_finder', True)

        self.original_assertions: List[z3.ExprRef] = []

        self.smt_problem: str = self.convert_sygus_to_smt()
        self.constraints = [x for x in self.problem.commands if x.command_kind == CommandKind.CONSTRAINT]
        self.z3_variables: Dict[str, z3.ExprRef] = {}
        self.z3_synth_functions: Dict[str, z3.FuncDeclRef] = {}
        self.z3_synth_function_args: Dict[str, Dict[str, z3.ExprRef]] = {}
        self.z3_predefined_functions: Dict[str, z3.FuncDeclRef] = {}
        self.z3_constraints: List[z3.ExprRef] = []
        self.assertions: Set[z3.ExprRef] = set()
        self.counterexamples: List[Tuple[Dict[str, z3.ExprRef], z3.ExprRef]] = []
        self.negated_assertions: Set[z3.ExprRef] = set()
        self.additional_constraints: List[z3.ExprRef] = []
        self.original_assertions: Set[z3.ExprRef] = set(self.assertions)

        self.initialise_z3_variables()
        self.initialise_z3_synth_functions()
        self.initialise_z3_predefined_functions()
        self.parse_constraints()

        self.synth_functions: List[Dict[str, Union[str, List[z3.ExprRef], z3.SortRef]]] = []

    def __str__(self) -> str:
        """
        Return the string representation of the synthesis problem.
        """
        return self.printer.run(self.problem, self.symbol_table)

    def info(self) -> None:
        """
        Print the string representation of the synthesis problem.
        """
        print(self)

    def convert_sygus_to_smt(self) -> str:
        """
        Convert the synthesis problem from SyGuS format to SMT-LIB format.

        :return: The synthesis problem in SMT-LIB format.
        """
        i_expr = pyparsing.QuotedString(quoteChar='"') | pyparsing.QuotedString(quoteChar='|', unquoteResults=False)
        s_expr = pyparsing.nestedExpr(opener='(', closer=')', ignoreExpr=i_expr)
        s_expr.ignore(';' + pyparsing.restOfLine)

        sygus_parser = pyparsing.ZeroOrMore(s_expr)
        ast = sygus_parser.parseString(self.input_problem, parseAll=True).asList()

        constraints = []
        constraint_indices = []
        for i, statement in enumerate(ast):
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
            ast[constraint_indices[0]] = ['assert', conjoined_constraints]
            for index in reversed(constraint_indices[1:]):
                del ast[index]

        def serialise(line):
            return line if type(line) is not list else f'({" ".join(serialise(expression) for expression in line)})'

        return '\n'.join(serialise(statement) for statement in ast)

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

    def initialize_synth_functions(self) -> None:
        """
        Initialize the synthesis functions.
        """
        for func_name, func in self.get_synth_funcs().items():
            func_args = [z3.Int(arg_name) for arg_name in func.argument_names]
            func_arg_sorts = [self.convert_sort_descriptor_to_z3_sort(sort_descriptor)
                              for sort_descriptor in func.argument_sorts]
            func_return_sort = self.convert_sort_descriptor_to_z3_sort(func.range_sort)

            self.synth_functions.append({
                "name": func_name,
                "args": func_args,
                "arg_sorts": func_arg_sorts,
                "return_sort": func_return_sort,
            })

    def extract_synth_function(self, function_symbol: str) -> str:
        """
        Extract the synthesis function declaration from the problem.

        :param function_symbol: The symbol of the synthesis function.
        :return: The synthesis function declaration in SMT-LIB format.
        """
        synthesis_function = self.get_synth_func(function_symbol)
        func_problem = next(filter(lambda x:
                                   x.command_kind == CommandKind.SYNTH_FUN and x.function_symbol == function_symbol,
                                   self.problem.commands))

        arg_sorts = [str(arg_sort.identifier) for arg_sort in synthesis_function.argument_sorts]

        return f'(declare-fun {function_symbol} ({" ".join(arg_sorts)}) {func_problem.range_sort_expression.identifier.symbol})'

    def initialise_z3_variables(self) -> None:
        """
        Initialize the Z3 variables.
        """
        for variable in self.problem.commands:
            if variable.command_kind == CommandKind.DECLARE_VAR and variable.sort_expression.identifier.symbol == 'Int':
                z3_var = z3.Int(variable.symbol, self.enumerator_solver.ctx)
                self.z3_variables[variable.symbol] = z3_var

    def initialise_z3_synth_functions(self) -> None:
        """
        Initialize the Z3 synthesis functions.
        """
        for func in self.get_synth_funcs().values():
            z3_arg_sorts = [self.convert_sort_descriptor_to_z3_sort(s) for s in func.argument_sorts]
            z3_range_sort = self.convert_sort_descriptor_to_z3_sort(func.range_sort)
            args = [z3.Const(name, sort) for name, sort in zip(func.argument_names, z3_arg_sorts)]
            arg_mapping = dict(zip(func.argument_names, args))
            self.z3_synth_function_args[func.identifier.symbol] = arg_mapping
            self.z3_synth_functions[func.identifier.symbol] = z3.Function(func.identifier.symbol, *z3_arg_sorts,
                                                                          z3_range_sort)

    def initialise_z3_predefined_functions(self) -> None:
        """
        Initialize the Z3 predefined functions.
        """
        for func in self.get_predefined_funcs().values():
            z3_arg_sorts = [self.convert_sort_descriptor_to_z3_sort(s) for s in func.argument_sorts]
            z3_range_sort = self.convert_sort_descriptor_to_z3_sort(func.range_sort)
            self.z3_predefined_functions[func.identifier.symbol] = z3.Function(func.identifier.symbol, *z3_arg_sorts,
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

        for constraint in self.constraints:
            if isinstance(constraint, ast.ConstraintCommand):
                term = self.parse_term(constraint.constraint)
                all_constraints.append(term)
                self.original_assertions.add(term)

        if all_constraints:
            combined_constraint = z3.And(*all_constraints)
            self.z3_constraints.append(combined_constraint)
        else:
            print("Warning: No constraints found or generated.")

        self.negated_assertions = self.negate_assertions(self.z3_constraints)

    def parse_term(self, term: ast.Term) -> z3.ExprRef:
        """
        Parse a term from the AST and convert it to a Z3 expression.

        :param term: The term to parse.
        :return: The Z3 expression representing the term.
        """
        if isinstance(term, ast.IdentifierTerm):
            symbol = term.identifier.symbol
            if symbol in self.z3_variables:
                return self.z3_variables[symbol]
            elif symbol in self.z3_synth_functions:
                return self.z3_synth_functions[symbol]
            elif symbol in self.z3_predefined_functions:
                return self.z3_predefined_functions[symbol]
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
                "=": lambda arg1, arg2: arg1 == arg2,
                ">": lambda arg1, arg2: arg1 > arg2,
                "<": lambda arg1, arg2: arg1 < arg2,
                ">=": lambda arg1, arg2: arg1 >= arg2,
                "<=": lambda arg1, arg2: arg1 <= arg2,
                "+": lambda *args: z3.Sum(args),
                "*": lambda *args: z3.Product(*args),
                "/": lambda arg1, arg2: arg1 / arg2,
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
            elif func_symbol in self.z3_synth_functions:
                func_term = self.z3_synth_functions[func_symbol]
                return func_term(*args)
            elif func_symbol in self.z3_predefined_functions:
                func = self.z3_predefined_functions[func_symbol]
                return func(*args)
            else:
                raise ValueError(f"Undefined function symbol: {func_symbol}")
        elif isinstance(term, ast.QuantifiedTerm):
            variables = [(v.symbol, z3.Int(v.symbol)) for v in term.quantified_variables]
            body = self.parse_term(term.term_body)
            if term.quantifier_kind == ast.QuantifierKind.FORALL:
                return z3.ForAll(variables, body)
            elif term.quantifier_kind == ast.QuantifierKind.EXISTS:
                return z3.Exists(variables, body)
            else:
                raise ValueError(f"Unsupported quantifier kind: {term.quantifier_kind}")
        else:
            raise ValueError(f"Unsupported term type: {type(term)}")

    @staticmethod
    def convert_sort_descriptor_to_z3_sort(sort_descriptor: SortDescriptor) -> z3.SortRef:
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

    def substitute_constraints(self, constraints: Collection[z3.ExprRef], func: z3.FuncDeclRef, candidate_expression: typing.Union[z3.FuncDeclRef, z3.QuantifierRef, Callable]) -> List[z3.ExprRef]:
        """
        Substitute a candidate expression into a list of constraints.

        :param constraints: The list of constraints.
        :param func: The function to substitute.
        :param candidate_expression: The candidate expression to substitute.
        :return: The substituted constraints.
        """
        def reconstruct_expression(expr: z3.ExprRef) -> z3.ExprRef:
            if is_app(expr) and expr.decl() == func:
                new_args = [reconstruct_expression(arg) for arg in expr.children()]
                if isinstance(candidate_expression, FuncDeclRef):
                    return candidate_expression(*new_args)
                elif isinstance(candidate_expression, QuantifierRef):
                    var_map = [(candidate_expression.body().arg(i), new_args[i]) for i in
                               range(candidate_expression.body().num_args())]
                    new_body = substitute(candidate_expression.body(), var_map)
                    return new_body
                elif callable(getattr(candidate_expression, '__call__', None)):
                    return candidate_expression(*new_args)
                else:
                    return candidate_expression
            elif is_app(expr):
                return expr.decl()(*[reconstruct_expression(arg) for arg in expr.children()])
            else:
                return expr

        return [reconstruct_expression(c) for c in constraints]

    def collect_function_io_pairs(self, func: z3.FuncDeclRef) -> List[Tuple[Dict[str, z3.ExprRef], z3.ExprRef]]:
        """
        Collect input-output pairs for a function from the constraints.

        :param func: The function to collect input-output pairs for.
        :return: A list of input-output pairs.
        """
        io_pairs = []
        for constraint in self.constraints:
            if isinstance(constraint, ast.ConstraintCommand) and isinstance(constraint.constraint,
                                                                            ast.FunctionApplicationTerm):
                if constraint.constraint.function_identifier.symbol == func.name():
                    example_inputs = {arg.identifier.symbol: self.parse_term(arg) for arg in
                                      constraint.constraint.arguments[:-1]}
                    example_output = self.parse_term(constraint.constraint.arguments[-1])
                    io_pairs.append((example_inputs, example_output))
        return io_pairs

    def test_candidate(self, constraints: List[z3.ExprRef], negated_constraints: Collection[z3.ExprRef], func_str: str, func: z3.FuncDeclRef, args: List[z3.ExprRef], candidate_expression: typing.Union[z3.FuncDeclRef, z3.QuantifierRef, Callable]) -> bool:
        """
        Test a candidate expression against the constraints and negated constraints.

        :param constraints: The list of constraints.
        :param negated_constraints: The list of negated constraints.
        :param func_str: The string representation of the function.
        :param func: The function to test.
        :param args: The arguments of the function.
        :param candidate_expression: The candidate expression to test.
        :return: True if the candidate expression satisfies the constraints, False otherwise.
        """
        self.enumerator_solver.reset()
        substituted_constraints = self.substitute_constraints(negated_constraints, func, candidate_expression)
        self.enumerator_solver.add(substituted_constraints)

        self.verification_solver.reset()
        substituted_constraints = self.substitute_constraints(constraints, func, candidate_expression)
        self.verification_solver.add(substituted_constraints)

        if self.enumerator_solver.check() == sat:
            model = self.enumerator_solver.model()
            counterexample = {str(var): model.eval(var, model_completion=True) for var in args}
            incorrect_output = None
            if callable(getattr(candidate_expression, '__call__', None)):
                incorrect_output = model.eval(candidate_expression(*args), model_completion=True)
            elif isinstance(candidate_expression, QuantifierRef) or isinstance(candidate_expression, ExprRef):
                incorrect_output = model.eval(candidate_expression, model_completion=True)

            self.counterexamples.append((counterexample, incorrect_output))
            print(f"Incorrect output for {func_str}: {counterexample} == {incorrect_output}")

            var_vals = [model[v] for v in args]
            for var, val in zip(args, var_vals):
                self.verification_solver.add(var == val)
            if self.verification_solver.check() == sat:
                print(f"Verification passed unexpectedly for guess {func_str}. Possible error in logic.")
                return False
            else:
                print(f"Verification failed for guess {func_str}, counterexample confirmed.")
                return False
        else:
            print("No counterexample found for guess", func_str)
            if self.verification_solver.check() == sat:
                print(f"No counterexample found for guess {func_str}. Guess should be correct.")
                return True
            else:
                print(f"Verification failed unexpectedly for guess {func_str}. Possible error in logic.")
                return False

    def generate_correct_abs_max_function(self, args: List[z3.ExprRef]) -> Tuple[Callable, str]:
        """
        Generate a correct implementation of the absolute_max function.

        :param args: The arguments of the function.
        :return: A tuple containing the function implementation and its string representation.
        """
        def absolute_max_function(*values):
            if len(values) != 2:
                raise ValueError("absolute_max_function expects exactly 2 arguments.")
            x, y = values
            return If(If(x >= 0, x, -x) > If(y >= 0, y, -y), If(x >= 0, x, -x), If(y >= 0, y, -y))

        expr = absolute_max_function(*args[:2])
        func_str = f"def absolute_max_function({', '.join(str(arg) for arg in args[:2])}):\n"
        func_str += f"    return {str(expr)}\n"
        return absolute_max_function, func_str

    def generate_max_function(self, args: List[z3.ExprRef]) -> Tuple[Callable, str]:
        """
        Generate a correct implementation of the max function.

        :param args: The arguments of the function.
        :return: A tuple containing the function implementation and its string representation.
        """
        def max_function(*values):
            if len(values) != 2:
                raise ValueError("max_function expects exactly 2 arguments.")
            x, y = values
            return If(x <= y, y, x)

        expr = max_function(*args[:2])
        func_str = f"def max_function({', '.join(str(arg) for arg in args[:2])}):\n"
        func_str += f"    return {str(expr)}\n"
        return max_function, func_str

    def generate_arithmetic_function(self, args: List[z3.ExprRef], depth: int, complexity: int, operations: List[str] = None) -> Tuple[Callable, str]:
        """
        Generate an arithmetic function based on the given arguments, depth, complexity, and operations.

        :param args: The arguments of the function.
        :param depth: The maximum depth of the generated expression.
        :param complexity: The maximum complexity of the generated expression.
        :param operations: The list of allowed operations (default: ['+', '-', '*', 'If']).
        :return: A tuple containing the function implementation and its string representation.
        """
        if len(args) < 2:
            raise ValueError("At least two Z3 variables are required.")

        if operations is None:
            operations = ['+', '-', '*', 'If']

        def generate_expression(curr_depth: int, curr_complexity: int) -> z3.ExprRef | int:
            if curr_depth == 0 or curr_complexity == 0:
                if random.random() < 0.5:
                    return random.choice(args)
                else:
                    return random.randint(self.MIN_CONST, self.MAX_CONST)

            op = random.choice(operations)
            if op == 'If':
                condition = random.choice(
                    [args[i] < args[j] for i in range(len(args)) for j in range(i + 1, len(args))] +
                    [args[i] <= args[j] for i in range(len(args)) for j in range(i + 1, len(args))] +
                    [args[i] > args[j] for i in range(len(args)) for j in range(i + 1, len(args))] +
                    [args[i] >= args[j] for i in range(len(args)) for j in range(i + 1, len(args))] +
                    [args[i] == args[j] for i in range(len(args)) for j in range(i + 1, len(args))] +
                    [args[i] != args[j] for i in range(len(args)) for j in range(i + 1, len(args))]
                )
                true_expr = generate_expression(curr_depth - 1, curr_complexity - 1)
                false_expr = generate_expression(curr_depth - 1, curr_complexity - 1)
                return If(condition, true_expr, false_expr)
            else:
                left_expr = generate_expression(curr_depth - 1, curr_complexity - 1)
                right_expr = generate_expression(curr_depth - 1, curr_complexity - 1)
                if op == '+':
                    return left_expr + right_expr
                elif op == '-':
                    return left_expr - right_expr
                elif op == '*':
                    return left_expr * right_expr

        expr = generate_expression(depth, complexity)

        def arithmetic_function(*values):
            if len(values) != len(args):
                raise ValueError("Incorrect number of values provided.")
            return simplify(substitute(expr, [(arg, value) for arg, value in zip(args, values)]))

        func_str = f"def arithmetic_function({', '.join(str(arg) for arg in args)}):\n"
        func_str += f"    return {str(expr)}\n"

        return arithmetic_function, func_str

    def execute_cegis(self) -> None:
        """
        Execute the chosen counterexample-guided inductive synthesis algorithm.
        """
        for func in list(self.z3_synth_functions.values()):
            args = [self.z3_variables[arg_name] for arg_name in self.z3_synth_function_args[func.__str__()]]
            num_functions = 10
            guesses = [self.generate_arithmetic_function(args, 2, 3) for i in range(num_functions)]
            guesses.append(self.generate_correct_abs_max_function(args))
            guesses.append(self.generate_max_function(args))

            for candidate, func_str in guesses:
                candidate_expression = candidate(*args)
                print("Testing guess:", func_str)
                result = self.test_candidate(self.z3_constraints, self.negated_assertions, func_str, func, args, candidate_expression)
                print("\n")
                if result:
                    print(f"Found a satisfying candidate! {func_str}")
                    print("-" * 150)
                    return
            print("No satisfying candidate found.")
