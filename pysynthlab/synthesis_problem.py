import itertools
from typing import List

from z3 import *
import pyparsing

from pysynthlab.helpers.parser.src import ast
from pysynthlab.helpers.parser.src.ast import Program, CommandKind
from pysynthlab.helpers.parser.src.resolution import FunctionKind, SortDescriptor
from pysynthlab.helpers.parser.src.symbol_table_builder import SymbolTableBuilder
from pysynthlab.helpers.parser.src.v1.parser import SygusV1Parser
from pysynthlab.helpers.parser.src.v1.printer import SygusV1ASTPrinter
from pysynthlab.helpers.parser.src.v2.parser import SygusV2Parser
from pysynthlab.helpers.parser.src.v2.printer import SygusV2ASTPrinter


class SynthesisProblem:
    MIN_CONST = -2
    MAX_CONST = 2
    pyparsing.ParserElement.enablePackrat()

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

        self.enumerator_solver = z3.Solver()
        self.enumerator_solver.set('smt.macro_finder', True)

        self.verification_solver = z3.Solver()
        self.verification_solver.set('smt.macro_finder', True)

        self.original_assertions = []

        self.smt_problem = self.convert_sygus_to_smt()
        self.constraints = [x for x in self.problem.commands if x.command_kind == CommandKind.CONSTRAINT]
        self.z3_variables = {}
        self.z3_synth_functions = {}
        self.z3_synth_function_args = {}
        self.z3_predefined_functions = {}
        self.z3_constraints = []
        self.assertions = set()
        self.counterexamples = set()
        self.negated_assertions = set()
        self.additional_constraints = []
        self.original_assertions = set(self.assertions)

        self.initialise_z3_variables()
        self.initialise_z3_synth_functions()
        self.initialise_z3_predefined_functions()
        self.parse_constraints()

        self.synth_functions = []

    def __str__(self) -> str:
        return self.printer.run(self.problem, self.symbol_table)

    def info(self) -> None:
        print(self)

    def convert_sygus_to_smt(self):
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

    def get_logic(self):
        return self.symbol_table.logic_name

    def get_synth_funcs(self):
        return self.symbol_table.synth_functions

    def get_predefined_funcs(self):
        return self.symbol_table.user_defined_functions

    def get_synth_func(self, symbol):
        return next(filter(lambda x:
                           x.function_kind == FunctionKind.SYNTH_FUN and x.identifier.symbol == symbol,
                           list(self.symbol_table.synth_functions.values())))

    def get_var_symbols(self):
        return [x.symbol for x in self.problem.commands if x.command_kind == CommandKind.DECLARE_VAR]

    def get_function_symbols(self):
        return [x.symbol for x in self.problem.commands if x.command_kind == CommandKind.DECLARE_FUN]

    def initialize_synth_functions(self):
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

    def extract_synth_function(self, function_symbol) -> str:
        synthesis_function = self.get_synth_func(function_symbol)
        func_problem = next(filter(lambda x:
                                   x.command_kind == CommandKind.SYNTH_FUN and x.function_symbol == function_symbol,
                                   self.problem.commands))

        arg_sorts = [str(arg_sort.identifier) for arg_sort in synthesis_function.argument_sorts]

        return f'(declare-fun {function_symbol} ({" ".join(arg_sorts)}) {func_problem.range_sort_expression.identifier.symbol})'

    def initialise_z3_variables(self):
        for variable in self.problem.commands:
            if variable.command_kind == CommandKind.DECLARE_VAR and variable.sort_expression.identifier.symbol == 'Int':
                z3_var = z3.Int(variable.symbol, self.enumerator_solver.ctx)
                self.z3_variables[variable.symbol] = z3_var

    def initialise_z3_synth_functions(self):
        for func in self.get_synth_funcs().values():
            z3_arg_sorts = [self.convert_sort_descriptor_to_z3_sort(s) for s in func.argument_sorts]
            z3_range_sort = self.convert_sort_descriptor_to_z3_sort(func.range_sort)
            args = [z3.Const(name, sort) for name, sort in zip(func.argument_names, z3_arg_sorts)]
            arg_mapping = dict(zip(func.argument_names, args))
            self.z3_synth_function_args[func.identifier.symbol] = arg_mapping
            self.z3_synth_functions[func.identifier.symbol] = z3.Function(func.identifier.symbol, *z3_arg_sorts,
                                                                          z3_range_sort)

    def initialise_z3_predefined_functions(self):
        for func in self.get_predefined_funcs().values():
            z3_arg_sorts = [self.convert_sort_descriptor_to_z3_sort(s) for s in func.argument_sorts]
            z3_range_sort = self.convert_sort_descriptor_to_z3_sort(func.range_sort)
            self.z3_predefined_functions[func.identifier.symbol] = z3.Function(func.identifier.symbol, *z3_arg_sorts,
                                                                               z3_range_sort)

    def generate_linear_integer_expressions(self, depth=0, size_limit=6, current_size=0):
        if depth == 0 or current_size >= size_limit:
            yield from [z3.IntVal(i) for i in range(self.MIN_CONST, self.MAX_CONST + 1)] + list(
                self.z3_variables.values())
            return

        for var in self.z3_variables.values():
            if current_size < size_limit:
                yield var
                yield var * z3.IntVal(-1)

            for expr in self.generate_linear_integer_expressions(depth - 1, size_limit, current_size + 1):
                yield var + expr
                yield var - expr
                yield expr - var

            for expr in self.generate_linear_integer_expressions(depth - 1, size_limit, current_size + 2):
                if current_size + 3 <= size_limit:
                    yield z3.If(var > expr, var, expr)
                    yield z3.If(var < expr, var, expr)
                    yield z3.If(var != expr, var, expr)

    def generate_candidate_expression(self, depth: int = 0) -> z3.ExprRef:
        expressions = self.generate_linear_integer_expressions(depth)
        for expr in itertools.islice(expressions, 200):  # limit breadth
            return expr

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
        else:
            print("Warning: No constraints found or generated.")

        self.negated_assertions = self.negate_assertions(self.z3_constraints)

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

    @staticmethod
    def convert_sort_descriptor_to_z3_sort(sort_descriptor: SortDescriptor) -> z3.SortRef:
        sort_symbol = sort_descriptor.identifier.symbol
        return {
            'Int': z3.IntSort(),
            'Bool': z3.BoolSort(),
        }.get(sort_symbol, None)

    def generate_candidate_functions(self, depth, size_limit=6, current_size=0):
        print(f"Generating at depth={depth}, size_limit={size_limit}, current_size={current_size}")
        # if depth == 1:
        #     print("Yielding the correct maximum function for testing")
        #     yield lambda args: z3.If(args[0] > args[1], args[0], args[1])
        #     return
        print("Yielding the correct maximum function for testing")
        yield lambda args: z3.If(args[0] > args[1], args[0], args[1])
        return

    # if depth == 0 or current_size >= size_limit:
    #     for i in range(self.MIN_CONST, self.MAX_CONST + 1):
    #         print(f"Yielding constant function for value {i}")
    #         yield lambda args, i=i: z3.IntVal(i)  # Explicitly wrap i with z3.IntVal to ensure correct type
    #     for var_name, var in self.z3_variables.items():
    #         print(f"Yielding identity function for variable {var_name}")
    #         yield lambda args, var=var: var
    #     return
    #
    # for var_name, var in self.z3_variables.items():
    #     index = list(self.z3_variables.keys()).index(var_name)
    #     print(f"Processing variable {var_name} at index {index}")
    #     if current_size < size_limit:
    #         yield lambda args, index=index: args[index]
    #         yield lambda args, index=index: -args[index]
    #
    #     for func in self.generate_candidate_functions(depth - 1, size_limit, current_size + 1):
    #         yield lambda args, index=index, func=func: args[index] + func(args)
    #         yield lambda args, index=index, func=func: args[index] - func(args)
    #         yield lambda args, index=index, func=func: func(args) - args[index]
    #
    #     for func in self.generate_candidate_functions(depth - 1, size_limit, current_size + 2):
    #         if current_size + 3 <= size_limit:
    #             yield lambda args, index=index, func=func: args[index] if args[index] > func(args) else func(args)
    #             yield lambda args, index=index, func=func: args[index] if args[index] < func(args) else func(args)
    #             yield lambda args, index=index, func=func: args[index] if args[index] != func(args) else func(args)

    def substitute_constraints(self, constraints, func, candidate_expression):
        def reconstruct_expression(expr):
            if is_app(expr) and expr.decl() == func:
                return candidate_expression(*[reconstruct_expression(arg) for arg in expr.children()])
            elif is_app(expr):
                return expr.decl()(*[reconstruct_expression(arg) for arg in expr.children()])
            else:
                return expr

        substituted_constraints = [reconstruct_expression(c) for c in constraints]
        return substituted_constraints

    def test_candidate(self, constraints, negated_constraints, name, func, args, candidate_expression):
        self.enumerator_solver.reset()
        substituted_constraints = self.substitute_constraints(negated_constraints, func, candidate_expression)
        self.enumerator_solver.add(substituted_constraints)

        self.verification_solver.reset()
        substituted_constraints = self.substitute_constraints(constraints, func, candidate_expression)
        self.verification_solver.add(substituted_constraints)

        if self.enumerator_solver.check() == sat:
            model = self.enumerator_solver.model()
            counterexample = {str(var): model.eval(var, model_completion=True) for var in args}
            incorrect_output = model.eval(candidate_expression(*args), model_completion=True)
            print(f"Incorrect output for {name}: {counterexample} == {incorrect_output}")

            if self.verification_solver.check() == sat:
                print(f"Verification passed unexpectedly for guess {name}. Possible error in logic.")
            else:
                print(f"Verification failed for guess {name}, counterexample confirmed.")
        else:
            print("No counterexample found for guess", name)
            if self.verification_solver.check() == sat:
                print(f"No counterexample found for guess {name}. Guess should be correct.")
            else:
                print(f"Verification failed unexpectedly for guess {name}. Possible error in logic.")

    def execute_cegis(self):

        def guess_a(x, y):
            return x + y

        def guess_b(x, y):
            return x - y

        def guess_c(x, y):
            return If(x <= y, y, x)

        def guess_d(x, y):
            return If(x > y, y, x)

        def guess_e(x, y):
            return IntVal(0)

        guesses = [
            (guess_e, "0"),
            (guess_a, "x+y"),
            (guess_b, "x-y"),
            (guess_c, "max(x, y)"),
            (guess_d, "min(x, y)"),
        ]
        func = list(self.z3_synth_functions.values())[0]
        args = [self.z3_variables[arg_name] for arg_name in self.z3_synth_function_args[func.__str__()]]
        for candidate, name in guesses:
            candidate_expression = Lambda(args, candidate(*args))
            print("Testing guess:", name)
            self.test_candidate(self.z3_constraints, self.negated_assertions, name, func, args, candidate_expression)

        print("-" * 50)