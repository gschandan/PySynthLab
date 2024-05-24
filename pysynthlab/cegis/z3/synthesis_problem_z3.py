import itertools
from typing import List, Dict, Set, Tuple, Any, Union, Optional
from z3 import *
import pyparsing
import random
from dataclasses import dataclass, field
from pysynthlab.helpers.parser.src import ast
from pysynthlab.helpers.parser.src.ast import Program, CommandKind, Term
from pysynthlab.helpers.parser.src.resolution import FunctionKind, SortDescriptor
from pysynthlab.helpers.parser.src.symbol_table_builder import SymbolTableBuilder
from pysynthlab.helpers.parser.src.v1.parser import SygusV1Parser
from pysynthlab.helpers.parser.src.v1.printer import SygusV1ASTPrinter
from pysynthlab.helpers.parser.src.v2.parser import SygusV2Parser
from pysynthlab.helpers.parser.src.v2.printer import SygusV2ASTPrinter


@dataclass
class SynthFunction:
    name: str
    args: List[Int]
    arg_sorts: List[Any]
    return_sort: Any


@dataclass
class SynthesisProblem:
    input_problem: str
    sygus_standard: int = 1
    options: Optional[Dict[str, Any]] = None
    problem: Program = field(init=False)
    symbol_table: Any = field(init=False)
    printer: Any = field(init=False)
    enumerator_solver: Solver = field(init=False, default_factory=Solver)
    verification_solver: Solver = field(init=False, default_factory=Solver)
    smt_problem: str = field(init=False)
    constraints: List[ast.Command] = field(init=False)
    z3_variables: Dict[str, Int] = field(default_factory=dict)
    z3_synth_functions: Dict[str, Any] = field(default_factory=dict)
    z3_synth_function_args: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    z3_predefined_functions: Dict[str, Any] = field(default_factory=dict)
    z3_constraints: List[ExprRef] = field(default_factory=list)
    assertions: Set[ExprRef] = field(default_factory=set)
    counterexamples: List[Tuple[Dict[str, Any], Any]] = field(default_factory=list)
    negated_assertions: Set[ExprRef] = field(default_factory=set)
    additional_constraints: List[ExprRef] = field(default_factory=list)
    synth_functions: List[SynthFunction] = field(default_factory=list)
    original_assertions: Set[ExprRef] = field(default_factory=set)

    MIN_CONST = -2
    MAX_CONST = 2
    pyparsing.ParserElement.enablePackrat()

    def __post_init__(self):
        self.options = self.options or {}
        self.parser = SygusV2Parser() if self.sygus_standard == 2 else SygusV1Parser()
        self.problem = self.parser.parse(self.input_problem)
        self.symbol_table = SymbolTableBuilder.run(self.problem)
        self.printer = (SygusV2ASTPrinter(self.symbol_table) if self.sygus_standard == 2
                        else SygusV1ASTPrinter(self.symbol_table, self.options))

        self.enumerator_solver.set('smt.macro_finder', True)
        self.verification_solver.set('smt.macro_finder', True)

        self.smt_problem = self.convert_sygus_to_smt()
        self.constraints = [x for x in self.problem.commands if x.command_kind == CommandKind.CONSTRAINT]

        self.initialise_z3_variables()
        self.initialise_z3_synth_functions()
        self.initialise_z3_predefined_functions()
        self.parse_constraints()

    def __str__(self) -> str:
        return self.printer.run(self.problem, self.symbol_table)

    def info(self) -> None:
        print(self)

    def convert_sygus_to_smt(self) -> str:
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

        def serialise(line: Any) -> str:
            return line if type(line) is not list else f'({" ".join(serialise(expression) for expression in line)})'

        return '\n'.join(serialise(statement) for statement in ast)

    def get_logic(self) -> str:
        return self.symbol_table.logic_name

    def get_synth_funcs(self) -> Dict[str, Any]:
        return self.symbol_table.synth_functions

    def get_predefined_funcs(self) -> Dict[str, Any]:
        return self.symbol_table.user_defined_functions

    def get_synth_func(self, symbol: str) -> Any:
        return next(filter(lambda x: x.function_kind == FunctionKind.SYNTH_FUN and x.identifier.symbol == symbol,
                           list(self.symbol_table.synth_functions.values())))

    def get_var_symbols(self) -> List[str]:
        return [x.symbol for x in self.problem.commands if x.command_kind == CommandKind.DECLARE_VAR]

    def get_function_symbols(self) -> List[str]:
        return [x.symbol for x in self.problem.commands if x.command_kind == CommandKind.DECLARE_FUN]

    def initialise_z3_variables(self) -> None:
        for variable in self.problem.commands:
            if variable.command_kind == CommandKind.DECLARE_VAR and variable.sort_expression.identifier.symbol == 'Int':
                z3_var = Int(variable.symbol, self.enumerator_solver.ctx)
                self.z3_variables[variable.symbol] = z3_var

    def initialise_z3_synth_functions(self) -> None:
        for func in self.get_synth_funcs().values():
            z3_arg_sorts = [self.convert_sort_descriptor_to_z3_sort(s) for s in func.argument_sorts]
            z3_range_sort = self.convert_sort_descriptor_to_z3_sort(func.range_sort)
            args = [Int(name, sort) for name, sort in zip(func.argument_names, z3_arg_sorts)]
            arg_mapping = dict(zip(func.argument_names, args))
            self.z3_synth_function_args[func.identifier.symbol] = arg_mapping
            self.z3_synth_functions[func.identifier.symbol] = z3.Function(func.identifier.symbol, *z3_arg_sorts, z3_range_sort)

    def initialise_z3_predefined_functions(self) -> None:
        for func in self.get_predefined_funcs().values():
            z3_arg_sorts = [self.convert_sort_descriptor_to_z3_sort(s) for s in func.argument_sorts]
            z3_range_sort = self.convert_sort_descriptor_to_z3_sort(func.range_sort)
            self.z3_predefined_functions[func.identifier.symbol] = z3.Function(func.identifier.symbol, *z3_arg_sorts, z3_range_sort)

    def parse_constraints(self) -> None:
        all_constraints = []

        for constraint in self.constraints:
            if isinstance(constraint, ast.ConstraintCommand):
                term = self.parse_term(constraint.constraint)
                all_constraints.append(term)
                self.original_assertions.add(term)

        if all_constraints:
            combined_constraint = z3.And(*all_constraints)
            self.z3_constraints.append(combined_constraint)

        self.negated_assertions = self.negate_assertions(self.z3_constraints)

    def parse_term(self, term: Term) -> ExprRef:
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
                return IntVal(int(literal.literal_value))
            elif literal.literal_kind == ast.LiteralKind.BOOLEAN:
                return Bool(literal.literal_value.lower() == "true")
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
            variables = [(v.symbol, Int(v.symbol)) for v in term.quantified_variables]
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
        sort_symbol = sort_descriptor.identifier.symbol
        return {
            'Int': z3.IntSort(),
            'Bool': z3.BoolSort(),
        }.get(sort_symbol, None)

    @staticmethod
    def negate_assertions(assertions: List[ExprRef]) -> List[ExprRef]:
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
                    raise ValueError(f"Unsupported assertion type: {assertion}")
        return negated_assertions

    def generate_linear_integer_expressions(self, depth: int = 0, size_limit: int = 6, current_size: int = 0):
        if depth == 0 or current_size >= size_limit:
            yield from [IntVal(i) for i in range(self.MIN_CONST, self.MAX_CONST + 1)] + list(self.z3_variables.values())
            return

        for var in self.z3_variables.values():
            if current_size < size_limit:
                yield var
                yield var * IntVal(-1)

            for expr in self.generate_linear_integer_expressions(depth - 1, size_limit, current_size + 1):
                yield var + expr
                yield var - expr
                yield expr - var

            for expr in self.generate_linear_integer_expressions(depth - 1, size_limit, current_size + 2):
                if current_size + 3 <= size_limit:
                    yield z3.If(var > expr, var, expr)
                    yield z3.If(var < expr, var, expr)
                    yield z3.If(var != expr, var, expr)

    def generate_candidate_expression(self, depth: int = 0) -> ExprRef:
        expressions = self.generate_linear_integer_expressions(depth)
        for expr in itertools.islice(expressions, 200):  # limit breadth
            return expr

    def substitute_constraints(self, constraints: List[ExprRef], func: Any, candidate_expression: Any) -> List[ExprRef]:
        def reconstruct_expression(expr: ExprRef) -> ExprRef:
            if is_app(expr) and expr.decl() == func:
                new_args = [reconstruct_expression(arg) for arg in expr.children()]
                if isinstance(candidate_expression, FuncDeclRef):
                    return candidate_expression(*new_args)
                elif isinstance(candidate_expression, QuantifierRef):
                    var_map = [(candidate_expression.body().arg(i), new_args[i]) for i in range(candidate_expression.body().num_args())]
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

    def collect_function_io_pairs(self, func: Any) -> List[Tuple[Dict[str, Any], Any]]:
        io_pairs = []
        for constraint in self.constraints:
            if isinstance(constraint, ast.ConstraintCommand) and isinstance(constraint.constraint, ast.FunctionApplicationTerm):
                if constraint.constraint.function_identifier.symbol == func.name():
                    example_inputs = {arg.identifier.symbol: self.parse_term(arg) for arg in constraint.constraint.arguments[:-1]}
                    example_output = self.parse_term(constraint.constraint.arguments[-1])
                    io_pairs.append((example_inputs, example_output))
        return io_pairs

    def test_candidate(self, constraints: List[ExprRef], negated_constraints: List[ExprRef], func_str: str, func: Any, args: List[ExprRef], candidate_expression: ExprRef) -> bool:

        self.enumerator_solver.reset()
        substituted_constraints = self.substitute_constraints(negated_constraints, func, candidate_expression)
        self.enumerator_solver.add(substituted_constraints)

        self.verification_solver.reset()
        substituted_constraints = self.substitute_constraints(constraints, func, candidate_expression)
        self.verification_solver.add(substituted_constraints)

        if self.enumerator_solver.check() == z3.sat:
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
            if self.verification_solver.check() == z3.sat:
                print(f"Verification passed unexpectedly for guess {func_str}. Possible error in logic.")
                return False
            else:
                print(f"Verification failed for guess {func_str}, counterexample confirmed.")
                return False
        else:
            print("No counterexample found for guess", func_str)
            if self.verification_solver.check() == z3.sat:
                print(f"No counterexample found for guess {func_str}. Guess should be correct.")
                return True
            else:
                print(f"Verification failed unexpectedly for guess {func_str}. Possible error in logic.")
                return False

    def generate_correct_abs_max_function(self, args: List[ExprRef]) -> Tuple[callable, str]:
        def absolute_max_function(*values: Any) -> Any:
            if len(values) != 2:
                raise ValueError("absolute_max_function expects exactly 2 arguments.")
            x, y = values
            return z3.If(z3.If(x >= 0, x, -x) > z3.If(y >= 0, y, -y), z3.If(x >= 0, x, -x), z3.If(y >= 0, y, -y))

        expr = absolute_max_function(*args[:2])
        func_str = f"def absolute_max_function({', '.join(str(arg) for arg in args[:2])}):\n"
        func_str += f"    return {str(expr)}\n"
        return absolute_max_function, func_str

    def generate_max_function(self, args: List[ExprRef]) -> Tuple[callable, str]:
        def max_function(*values: Any) -> Any:
            if len(values) != 2:
                raise ValueError("max_function expects exactly 2 arguments.")
            x, y = values
            return z3.If(x <= y, y, x)

        expr = max_function(*args[:2])
        func_str = f"def max_function({', '.join(str(arg) for arg in args[:2])}):\n"
        func_str += f"    return {str(expr)}\n"
        return max_function, func_str

    def generate_arithmetic_function(self, args: List[ExprRef], depth: int, complexity: int, operations: Optional[List[str]] = None) -> Tuple[callable, str]:
        if len(args) < 2:
            raise ValueError("At least two Z3 variables are required.")

        if operations is None:
            operations = ['+', '-', '*', 'If']

        def generate_expression(curr_depth: int, curr_complexity: int) -> Any:
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
                return z3.If(condition, true_expr, false_expr)
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

        def arithmetic_function(*values: Any) -> Any:
            if len(values) != len(args):
                raise ValueError("Incorrect number of values provided.")
            return simplify(substitute(expr, [(arg, value) for arg, value in zip(args, values)]))

        func_str = f"def arithmetic_function({', '.join(str(arg) for arg in args)}):\n"
        func_str += f"    return {str(expr)}\n"

        return arithmetic_function, func_str

    def execute_cegis(self) -> None:
        for func in list(self.z3_synth_functions.values()):
            args = [self.z3_variables[arg_name] for arg_name in self.z3_synth_function_args[func.__str__()]]
            num_functions = 10
            guesses = [self.generate_arithmetic_function(args, 2, 3) for _ in range(num_functions)]
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
