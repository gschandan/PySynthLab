import z3

from pysynthlab.helpers.parser.src import symbol_table_builder
from pysynthlab.helpers.parser.src.ast import Program, CommandKind
from pysynthlab.helpers.parser.src.resolution import SymbolTable, FunctionKind
from pysynthlab.helpers.parser.src.v1.parser import SygusV1Parser
from pysynthlab.helpers.parser.src.v1.printer import SygusV1ASTPrinter
from pysynthlab.helpers.parser.src.v2.parser import SygusV2Parser
from pysynthlab.helpers.parser.src.v2.printer import SygusV2ASTPrinter


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
        self.problem: Program = self.parser.parse(problem)
        self.symbol_table: SymbolTable = symbol_table_builder.SymbolTableBuilder.run(self.problem)
        self.printer: SygusV2ASTPrinter | SygusV1ASTPrinter = SygusV2ASTPrinter(self.symbol_table) \
            if sygus_standard == 2 \
            else SygusV1ASTPrinter(self.symbol_table, options)
        self.solver = solver
        self.commands = [x for x in self.problem.commands]
        self.z3variables = []
        self.constraints = [x for x in self.problem.commands if x.command_kind == CommandKind.CONSTRAINT]

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

    def add_variables(self):

        for variable in [x for x in self.problem.commands if x.command_kind == CommandKind.DECLARE_VAR]:
            if variable.__getattribute__('sort_expression').identifier.symbol == 'Int':
                self.z3variables.append(z3.Int(variable.symbol))
            print(variable.symbol, variable.__getattribute__('sort_expression').identifier.symbol == 'Int')

    def setup_solver(self):

        self.add_variables()
