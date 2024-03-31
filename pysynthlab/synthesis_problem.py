import itertools

import z3
import pyparsing
from pysynthlab.helpers.parser.src import symbol_table_builder
from pysynthlab.helpers.parser.src.ast import Program, CommandKind, GrammarTermKind
from pysynthlab.helpers.parser.src.resolution import SymbolTable, FunctionKind, SortDescriptor
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
        self.options: object = options
        self.sygus_standard: int = sygus_standard
        self.parser: SygusV1Parser | SygusV2Parser = SygusV2Parser() if sygus_standard == 2 else SygusV1Parser()
        self.input_problem: str = problem
        self.problem: Program = self.parser.parse(problem)
        self.symbol_table: SymbolTable = symbol_table_builder.SymbolTableBuilder.run(self.problem)
        self.printer: SygusV2ASTPrinter | SygusV1ASTPrinter = SygusV2ASTPrinter(self.symbol_table) \
            if sygus_standard == 2 \
            else SygusV1ASTPrinter(self.symbol_table, options)

        self.enumerator_solver = z3.Solver()
        self.enumerator_solver.push()
        self.verification_solver = z3.Solver()
        self.verification_solver.push()

        self.commands = [x for x in self.problem.commands]
        self.constraints = [x for x in self.problem.commands if x.command_kind == CommandKind.CONSTRAINT]
        self.smt_problem = self.convert_sygus_to_smt()
        self.synthesis_functions = []

        self.z3variables = {}
        self.z3function_definitions = []
        self.z3_synth_functions = {}
        self.z3_predefined_functions = {}
        self.z3_constraints = []

        self.initialise_z3_variables()
        self.initialise_z3_synth_functions()
        self.initialise_z3_predefined_functions()

        self.assertions = set()
        self.counterexamples = set()
        self.negated_assertions = set()
        self.additional_constraints = []
        self.original_assertions = []

        # todo: refactor for problems with more than one func to synthesisise
        self.func_name, self.z3_func = list(self.z3_synth_functions.items())[0]
        self.func_to_synthesise = self.get_synth_func(self.func_name)
        self.func_args = [z3.Int(name) for name in self.func_to_synthesise.argument_names]
        self.arg_sorts = [self.convert_sort_descriptor_to_z3_sort(sort_descriptor) for sort_descriptor in
                          self.func_to_synthesise.argument_sorts]

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
                self.z3variables[variable.symbol] = z3_var

    def initialise_z3_synth_functions(self):
        for func in self.get_synth_funcs().values():
            z3_arg_sorts = [self.convert_sort_descriptor_to_z3_sort(s) for s in func.argument_sorts]
            z3_range_sort = self.convert_sort_descriptor_to_z3_sort(func.range_sort)
            self.z3_synth_functions[func.identifier.symbol] = z3.Function(func.identifier.symbol, *z3_arg_sorts,
                                                                          z3_range_sort)

    def initialise_z3_predefined_functions(self):
        for func in self.get_predefined_funcs().values():
            z3_arg_sorts = [self.convert_sort_descriptor_to_z3_sort(s) for s in func.argument_sorts]
            z3_range_sort = self.convert_sort_descriptor_to_z3_sort(func.range_sort)
            self.z3_predefined_functions[func.identifier.symbol] = z3.Function(func.identifier.symbol, *z3_arg_sorts,
                                                                               z3_range_sort)

    def generate_linear_integer_expressions(self, depth, size_limit=6, current_size=0):
        if depth == 0 or current_size >= size_limit:
            yield from [z3.IntVal(i) for i in range(self.MIN_CONST, self.MAX_CONST + 1)] + list(
                self.z3variables.values())
            return

        for var in self.z3variables.values():
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

    def generate_candidate_functions(self, depth, size_limit=6, current_size=0):
        if depth == 0 or current_size >= size_limit:
            yield from ([lambda *args: i for i in range(self.MIN_CONST, self.MAX_CONST + 1)] +
                        [lambda *args: var for var in self.z3variables.keys()])
            return

        for var_name, var in self.z3variables.items():
            if current_size < size_limit:
                yield lambda *args: args[list(self.z3variables.keys()).index(var_name)]
                yield lambda *args: -args[list(self.z3variables.keys()).index(var_name)]

            for func in self.generate_candidate_functions(depth - 1, size_limit, current_size + 1):
                yield lambda *args: args[list(self.z3variables.keys()).index(var_name)] + func(*args)
                yield lambda *args: args[list(self.z3variables.keys()).index(var_name)] - func(*args)
                yield lambda *args: func(*args) - args[list(self.z3variables.keys()).index(var_name)]

            for func in self.generate_candidate_functions(depth - 1, size_limit, current_size + 2):
                if current_size + 3 <= size_limit:
                    yield lambda *args: args[list(self.z3variables.keys()).index(var_name)] if args[list(
                        self.z3variables.keys()).index(var_name)] > func(*args) else func(*args)
                    yield lambda *args: args[list(self.z3variables.keys()).index(var_name)] if args[list(
                        self.z3variables.keys()).index(var_name)] < func(*args) else func(*args)
                    yield lambda *args: args[list(self.z3variables.keys()).index(var_name)] if args[list(
                        self.z3variables.keys()).index(var_name)] != func(*args) else func(*args)


    def check_counterexample(self, model):
        for constraint in self.original_assertions:
            if not model.eval(constraint, model_completion=True):
                return {str(arg): model[arg] for arg in self.func_args}
        return None

    def get_additional_constraints(self, counterexample):
        constraints = [var != counterexample[var.__str__()] for var in self.func_args]
        return z3.And(*constraints)

    def generate_candidate_expression(self, depth=0):
        expressions = self.generate_linear_integer_expressions(depth)
        for expr in itertools.islice(expressions, 200):  # limit breadth
            return expr

    @staticmethod
    def negate_assertions(assertions):

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

    @staticmethod
    def convert_sort_descriptor_to_z3_sort(sort_descriptor: SortDescriptor):
        sort_symbol = sort_descriptor.identifier.symbol
        return {
            'Int': z3.IntSort(),
            'Bool': z3.BoolSort(),
        }.get(sort_symbol, None)


