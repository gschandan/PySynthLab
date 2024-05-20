import typing
from dataclasses import dataclass
from typing import List, Dict, Tuple, Callable, Union
from z3 import *
import pyparsing
import random
from pysynthlab.helpers.parser.src import ast
from pysynthlab.helpers.parser.src.ast import Program, CommandKind
from pysynthlab.helpers.parser.src.resolution import FunctionKind, SortDescriptor, SymbolTable
from pysynthlab.helpers.parser.src.symbol_table_builder import SymbolTableBuilder
from pysynthlab.helpers.parser.src.v1.parser import SygusV1Parser
from pysynthlab.helpers.parser.src.v1.printer import SygusV1ASTPrinter
from pysynthlab.helpers.parser.src.v2.parser import SygusV2Parser
from pysynthlab.helpers.parser.src.v2.printer import SygusV2ASTPrinter


@dataclass
class SynthFunctionSpec:
    name: str
    args: List[z3.ArithRef]
    arg_sorts: List[z3.SortRef]
    return_sort: z3.SortRef


class ConstraintParser:
    def __init__(self, problem: Program, symbol_table:  SymbolTable):
        self.problem = problem
        self.symbol_table = symbol_table
        self.z3_variables = {}
        self.z3_synth_functions = {}
        self.z3_predefined_functions = {}
        self.z3_constraints = []

        self.initialise_z3_variables()
        self.initialise_z3_synth_functions()
        self.initialise_z3_predefined_functions()
        self.parse_constraints()

    def initialise_z3_variables(self):
        for variable in self.problem.commands:
            if variable.command_kind == CommandKind.DECLARE_VAR and variable.sort_expression.identifier.symbol == 'Int':
                z3_var = z3.Int(variable.symbol)
                self.z3_variables[variable.symbol] = z3_var

    def initialise_z3_synth_functions(self):
        for func in self.symbol_table.synth_functions.values():
            z3_arg_sorts = [self.convert_sort_descriptor_to_z3_sort(s) for s in func.argument_sorts]
            z3_range_sort = self.convert_sort_descriptor_to_z3_sort(func.range_sort)
            self.z3_synth_functions[func.identifier.symbol] = z3.Function(func.identifier.symbol, *z3_arg_sorts,
                                                                          z3_range_sort)

    def initialise_z3_predefined_functions(self):
        for func in self.symbol_table.user_defined_functions.values():
            z3_arg_sorts = [self.convert_sort_descriptor_to_z3_sort(s) for s in func.argument_sorts]
            z3_range_sort = self.convert_sort_descriptor_to_z3_sort(func.range_sort)
            self.z3_predefined_functions[func.identifier.symbol] = z3.Function(func.identifier.symbol, *z3_arg_sorts,
                                                                               z3_range_sort)

    @staticmethod
    def convert_sort_descriptor_to_z3_sort(sort_descriptor: SortDescriptor) -> z3.SortRef:
        sort_symbol = sort_descriptor.identifier.symbol
        return {
            'Int': z3.IntSort(),
            'Bool': z3.BoolSort(),
        }.get(sort_symbol, None)

    def parse_constraints(self) -> None:
        all_constraints = []

        for constraint in self.problem.commands:
            if isinstance(constraint, ast.ConstraintCommand):
                term = self.parse_term(constraint.constraint)
                all_constraints.append(term)

        if all_constraints:
            combined_constraint = z3.And(*all_constraints)
            self.z3_constraints.append(combined_constraint)
        else:
            print("Warning: No constraints found or generated.")

    def parse_term(self, term: ast.Term) -> z3.ExprRef:
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


class CandidateGenerator:
    MIN_CONST = -2
    MAX_CONST = 2

    def __init__(self, synth_functions: Dict[str, SynthFunctionSpec]):
        self.synth_functions = synth_functions

    def generate_correct_abs_max_function(self, args: List[z3.ArithRef]) -> Tuple[Callable, str]:
        def absolute_max_function(*values):
            if len(values) != 2:
                raise ValueError("absolute_max_function expects exactly 2 arguments.")
            x, y = values
            return If(If(x >= 0, x, -x) > If(y >= 0, y, -y), If(x >= 0, x, -x), If(y >= 0, y, -y))

        expr = absolute_max_function(*args[:2])
        func_str = f"def absolute_max_function({', '.join(str(arg) for arg in args[:2])}):\n"
        func_str += f"    return {str(expr)}\n"
        return absolute_max_function, func_str

    def generate_max_function(self, args: List[z3.ArithRef]) -> Tuple[Callable, str]:
        def max_function(*values):
            if len(values) != 2:
                raise ValueError("max_function expects exactly 2 arguments.")
            x, y = values
            return If(x <= y, y, x)

        expr = max_function(*args[:2])
        func_str = f"def max_function({', '.join(str(arg) for arg in args[:2])}):\n"
        func_str += f"    return {str(expr)}\n"
        return max_function, func_str

    def generate_arithmetic_function(self, args: List[z3.ArithRef], depth: int, complexity: int,
                                     operations: List[str] = None) -> Tuple[Callable, str]:
        if len(args) < 2:
            raise ValueError("At least two Z3 variables are required.")

        if operations is None:
            operations = ['+', '-', '*', 'If']

        def generate_expression(curr_depth, curr_complexity):
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

    def generate_candidates(self, func_spec: SynthFunctionSpec) -> List[Tuple[Callable, str]]:
        args = func_spec.args
        num_functions = 10
        candidates = [self.generate_arithmetic_function(args, 2, 3) for _ in range(num_functions)]
        candidates.append(self.generate_correct_abs_max_function(args))
        candidates.append(self.generate_max_function(args))
        return candidates


class CegisSolver:
    def __init__(self, problem: Program, synth_functions: Dict[str, SynthFunctionSpec], constraints: List[z3.ExprRef]):
        self.problem = problem
        self.synth_functions = synth_functions
        self.constraints = constraints
        self.counterexamples = []

        self.enumerator_solver = z3.Solver()
        self.enumerator_solver.set('smt.macro_finder', True)

        self.verification_solver = z3.Solver()
        self.verification_solver.set('smt.macro_finder', True)

    @staticmethod
    def negate_assertions(assertions: List[z3.ExprRef]) -> List[z3.ExprRef]:
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

    def substitute_constraints(self, constraints: List[ExprRef], func_name: str, candidate_expression: typing.Union[ExprRef, QuantifierRef, Callable]) -> List[ExprRef]:
        def reconstruct_expression(expr):
            if is_app(expr) and expr.decl().name() == func_name:
                new_args = [reconstruct_expression(arg) for arg in expr.children()]
                if isinstance(candidate_expression, FuncDeclRef):
                    return candidate_expression(*new_args)
                elif isinstance(candidate_expression, QuantifierRef):
                    var_map = [(candidate_expression.body().arg(i), new_args[i]) for i in
                               range(candidate_expression.body().num_args())]
                    new_body = substitute(candidate_expression.body(), var_map)
                    return new_body
                elif callable(candidate_expression):
                    return candidate_expression(*new_args)
                else:
                    return candidate_expression
            elif is_app(expr):
                return expr.decl()(*[reconstruct_expression(arg) for arg in expr.children()])
            else:
                return expr

        return [reconstruct_expression(c) for c in constraints]

    def test_candidate(self, func_spec: SynthFunctionSpec, func_str: str, candidate_expression: typing.Union[z3.ExprRef, z3.QuantifierRef, Callable]) -> bool:
        negated_constraints = self.negate_assertions(self.constraints)

        self.enumerator_solver.reset()
        substituted_constraints = self.substitute_constraints(negated_constraints, func_spec.name, candidate_expression)
        self.enumerator_solver.add(substituted_constraints)

        self.verification_solver.reset()
        substituted_constraints = self.substitute_constraints(self.constraints, func_spec.name, candidate_expression)
        self.verification_solver.add(substituted_constraints)

        if self.enumerator_solver.check() == sat:
            model = self.enumerator_solver.model()
            counterexample = {str(var): model.eval(var, model_completion=True) for var in func_spec.args}
            incorrect_output = None
            if callable(getattr(candidate_expression, '__call__', None)):
                incorrect_output = model.eval(candidate_expression(*func_spec.args), model_completion=True)
            elif isinstance(candidate_expression, QuantifierRef) or isinstance(candidate_expression, ExprRef):
                incorrect_output = model.eval(candidate_expression, model_completion=True)

            self.counterexamples.append((counterexample, incorrect_output))
            print(f"Incorrect output for {func_str}: {counterexample} == {incorrect_output}")

            var_vals = [model[v] for v in func_spec.args]
            for var, val in zip(func_spec.args, var_vals):
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

    def execute_cegis(self):
        candidate_generator = CandidateGenerator(self.synth_functions)

        for func_spec in self.synth_functions.values():
            candidates = candidate_generator.generate_candidates(func_spec)

            for candidate, func_str in candidates:
                candidate_expression = candidate(*func_spec.args)
                print("Testing guess:", func_str)
                result = self.test_candidate(func_spec, func_str, candidate_expression)
                print("\n")
                if result:
                    print(f"Found a satisfying candidate! {func_str}")
                    print("-" * 150)
                    return
            print("No satisfying candidate found.")


class SynthesisProblem:
    def __init__(self, problem: str, sygus_standard: int = 1, options: object = None):
        if options is None:
            options = {}

        self.input_problem: str = problem
        self.options = options
        self.sygus_standard = sygus_standard
        self.parser = SygusV2Parser() if sygus_standard == 2 else SygusV1Parser()
        self.problem: Program = self.parser.parse(problem)
        self.symbol_table = SymbolTableBuilder.run(self.problem)
        self.printer = SygusV2ASTPrinter(self.symbol_table) if sygus_standard == 2 else SygusV1ASTPrinter(
            self.symbol_table, options)

        self.constraint_parser = ConstraintParser(self.problem, self.symbol_table)
        self.synth_functions = self.build_synth_functions()

    def __str__(self) -> str:
        return self.printer.run(self.problem, self.symbol_table)

    def info(self) -> None:
        print(self)

    def build_synth_functions(self) -> Dict[str, SynthFunctionSpec]:
        synth_functions = {}
        for func_name, func in self.symbol_table.synth_functions.items():
            func_args = [z3.Int(arg_name) for arg_name in func.argument_names]
            func_arg_sorts = [self.constraint_parser.convert_sort_descriptor_to_z3_sort(sort_descriptor) for
                              sort_descriptor in func.argument_sorts]
            func_return_sort = self.constraint_parser.convert_sort_descriptor_to_z3_sort(func.range_sort)
            synth_functions[func_name] = SynthFunctionSpec(func_name, func_args, func_arg_sorts, func_return_sort)
        return synth_functions

    def execute_cegis(self):
        cegis_solver = CegisSolver(self.problem, self.synth_functions, self.constraint_parser.z3_constraints)
        cegis_solver.execute_cegis()
