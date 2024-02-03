import z3
import pyparsing
from pysynthlab.helpers.parser.src import symbol_table_builder
from pysynthlab.helpers.parser.src.ast import Program, CommandKind
from pysynthlab.helpers.parser.src.resolution import SymbolTable, FunctionKind
from pysynthlab.helpers.parser.src.v1.parser import SygusV1Parser
from pysynthlab.helpers.parser.src.v1.printer import SygusV1ASTPrinter
from pysynthlab.helpers.parser.src.v2.parser import SygusV2Parser
from pysynthlab.helpers.parser.src.v2.printer import SygusV2ASTPrinter
from functools import lru_cache


class SynthesisProblem:
    """
        A class representing a synthesis problem.
        ...
        Attributes
        ----------
        options : object
            Additional options for the synthesis problem.
        sygus_standard : int
            The selected SyGuS-IF standard version.
        parser : SygusParser (SygusV1Parser or SygusV2Parser)
            The SyGuS parser based on the chosen standard.
        problem : Program
            The parsed synthesis problem.
        symbol_table : SymbolTable
            The symbol table built from the parsed problem.
        printer : SygusASTPrinter (SygusV1ASTPrinter or SygusV2ASTPrinter)
            The AST (Abstract Syntax Tree) printer based on the chosen standard.
        ...
        Methods
        -------
        info():
            Prints the synthesis problem to the console
        """

    def __init__(self, problem: str, solver: z3.Solver, sygus_standard: int = 1, options: object = None):
        """
        Initialize a SynthesisProblem instance with the provided parameters.

        Parameters
        ----------
        problem : str
            The synthesis problem to be parsed.
        sygus_standard : int, optional
            The SyGuS-IF standard version (1 or 2). Default is 2.
        options : object, optional
            Additional options for version 1 (e.g., 'no-unary-minus'). Default is None.

        Examples
        --------
        >> problem = SynthesisProblem("(set-logic LIA)\n(synth-fun f ((x Int) (y Int)) Int)\n...", sygus_standard=1)
        >> print(problem.problem)
        (set-logic LIA)
        (synth-fun f ((x Int) (y Int)) Int)
        ...

        """
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
        self.solver = solver
        self.commands = [x for x in self.problem.commands]
        self.z3variables = {}
        self.z3function_definitions = []
        self.constraints = [x for x in self.problem.commands if x.command_kind == CommandKind.CONSTRAINT]

        pyparsing.ParserElement.enablePackrat()
        self.smt_problem = self.convert_sygus_to_smt()

        self.initialise_variables()
        self.z3functions = []
        self.initialise_z3_functions()

    def __str__(self) -> str:
        """
        Returns a string representation of the synthesis problem.
        :returns: str: A string representation of the synthesis problem.
        """
        return self.printer.run(self.problem, self.symbol_table)

    def info(self) -> None:
        """ Prints the synthesis problem to the console
        :returns: None
        """
        print(self)

    def convert_sygus_to_smt(self):
        i_expr = pyparsing.QuotedString(quoteChar='"') | pyparsing.QuotedString(quoteChar='|', unquoteResults=False)
        s_expr = pyparsing.nestedExpr(opener='(', closer=')', ignoreExpr=i_expr)
        s_expr.ignore(';' + pyparsing.restOfLine)

        sygus_parser = pyparsing.ZeroOrMore(s_expr)
        ast = sygus_parser.parseString(self.input_problem, parseAll=True).asList()

        for statement in ast:
            if statement[0] == 'constraint':
                statement[0] = 'assert'
            elif statement[0] == 'check-synth':
                statement[0] = 'check-sat'
            elif statement[0] == 'synth-fun':
                statement[0] = 'declare-fun'
                statement[2] = [var_decl[1] for var_decl in statement[2]]

        def serialise(line):
            return line if type(line) is not list else f'({" ".join(serialise(expression) for expression in line)})'

        return '\n'.join(serialise(statement) for statement in ast)

    def get_logic(self):
        return self.symbol_table.logic_name

    def get_synth_funcs(self):
        return self.symbol_table.synth_functions

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

    def initialise_variables(self):
        for variable in self.problem.commands:
            if variable.command_kind == CommandKind.DECLARE_VAR and variable.sort_expression.identifier.symbol == 'Int':
                z3_var = z3.Int(variable.symbol, self.solver.ctx)
                self.z3variables[variable.symbol] = z3_var

    def initialise_z3_functions(self):
        self.z3functions = {func.identifier.symbol: self.create_z3_function(func) for func in
                            self.get_synth_funcs().values()}

    def generate_linear_integer_expressions(self, depth):
        """Generates linear integer expressions up to a given depth, yielding candidate expressions"""
        if depth == 0:
            yield from [z3.IntVal(i) for i in range(-10, 5)]
        else:
            for var_name, var in self.z3variables.items():
                for expr in self.generate_linear_integer_expressions(depth - 1):
                    # Arithmetic
                    yield var + expr
                    yield var - expr
                    yield var * z3.IntVal(2)
                    yield var * z3.IntVal(-1)
                    # Conditional
                    for other_expr in self.generate_linear_integer_expressions(depth - 1):
                        yield z3.If(var > other_expr, var, other_expr)
                        yield z3.If(var < other_expr, var, other_expr)
                        yield z3.If(var == other_expr, var, expr)
                        yield z3.If(var != other_expr, var, expr)

    @lru_cache(maxsize=None)
    def generate_linear_integer_expressions_v2(self, depth):
        """
        Generates linear integer expressions up to a given depth using memoisation to reduce computation.
        Performance isn't great so commented out other operations
        """
        if depth == 0:
            return [z3.IntVal(i) for i in range(-20, 20)]

        expressions = []
        for var_name, var in self.z3variables.items():
            for expr in self.generate_linear_integer_expressions(depth - 1):
                expressions.extend([
                    var + expr,
                    var - expr,
                    # var * z3.IntVal(2),
                    var * z3.IntVal(-1),
                ])

                # generate conditional expressions only once per pair to reduce computation
                # for other_expr in (expr2 for expr2 in self.generate_linear_integer_expressions(depth - 1) if
                #                    expr2 is not expr):
                #     expressions.extend([
                #         z3.If(var > other_expr, var, other_expr),
                #         z3.If(var >= other_expr, var, other_expr),
                #         z3.If(var >= other_expr, other_expr, var),
                #         z3.If(var < other_expr, var, other_expr),
                #         z3.If(var <= other_expr, var, other_expr),
                #         z3.If(var <= other_expr, other_expr, var),
                #         z3.If(var == other_expr, var, expr),
                #         z3.If(var != other_expr, var, expr),
                #     ])
        return iter(expressions)

    def generate_linear_integer_expressions_v3(self, depth, variables=None):
        if variables is None:
            variables = list(self.z3variables.values())

        # Base case: yield integer constants and variables directly without recursion as getting recursion errors in v1
        if depth == 0:
            for i in range(-6, 5):
                yield z3.IntVal(i)
            for var in variables:
                yield var
        else:
            # Generate expressions from previous depth
            previous_expressions = list(self.generate_linear_integer_expressions_v3(depth - 1, variables))
            for expr in previous_expressions:
                for var in variables:
                    yield var + expr
                    yield var - expr
                    yield expr - var
                    yield var * z3.IntVal(2)
                    yield var * z3.IntVal(-1)
                    # Generate combinations + avoiding recursion by using previously generated expressions
                    for other_expr in previous_expressions:
                        if expr.sort() == other_expr.sort():  # check sorts to prevent invalid operations
                            yield expr + other_expr
                            yield expr - other_expr
                            yield z3.If(var > other_expr, expr, other_expr)
                            yield z3.If(var < other_expr, expr, other_expr)

    def generate_linear_integer_expressions_v4(self, depth, size_limit, variables=None, current_size=0):

        if variables is None:
            variables = list(self.z3variables.values())

        if depth == 0 or current_size >= size_limit:
            yield from [z3.IntVal(i) for i in range(-10, 11)]
        else:
            for var_name, var in variables.items():
                if current_size < size_limit:
                    yield var

                for expr in self.generate_linear_integer_expressions_v4(depth - 1, size_limit,variables, current_size + 1):
                    # impose size limit
                    if current_size + 1 < size_limit:
                        yield var + expr
                        yield var - expr
                        yield var * z3.IntVal(2)
                        yield var * z3.IntVal(-1)

                        for other_expr in self.generate_linear_integer_expressions_v4(depth - 1, size_limit, variables,  current_size + 2):
                            if current_size + 2 < size_limit:
                                yield z3.If(var > other_expr, var, other_expr)
                                yield z3.If(var < other_expr, var, other_expr)

    @staticmethod
    def create_z3_function(func_descriptor):
        arg_sorts = [z3.IntSort() for arg_sort in func_descriptor.argument_sorts if arg_sort == 'Int']
        return z3.Function(func_descriptor.identifier.symbol, *arg_sorts, z3.IntSort())
