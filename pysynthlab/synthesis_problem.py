from helpers.parser.src.v1.parser import SygusV1Parser
from helpers.parser.src.v2.parser import SygusV2Parser
from helpers.parser.src import symbol_table_builder
from helpers.parser.src.ast import Program
from helpers.parser.src.resolution import SymbolTable
from helpers.parser.src.v1.printer import SygusV1ASTPrinter
from helpers.parser.src.v2.printer import SygusV2ASTPrinter


class SynthesisProblem:
    """
        A class representing a synthesis problem.
        Attributes
        ----------
        ...
        Methods
        -------
        info():
            Prints the synthesis problem to the console
        get_logic
            get the logic for the synthesis problem e.g. LIA (Linear Integer Arithmetic)
        get_synthesis_function : str
            get the function to be synthesised
        get_variables : dict[str,str]
            get the declared variables: <name,type>
        get_constraints : list[str]
            constraints to specify a desired property of the synthesised function
        """

    def __init__(self, problem: str, sygus_standard: int = 2, options: object = None):
        """
        Constructs all the necessary attributes for the synthesis problem.

        Parameters
        ----------
            problem: str
                synthesis problem to be parsed
            sygus_standard: int
                v1 or v2 SyGuS-IF standard
            options: object
                additional options if V1 problem e.g. no-unary-minus
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
        pass

    def get_synthesis_function(self):
        pass

    def get_variables(self):
        pass

    def get_constraints(self):
        pass

    def enumerator(self):
        pass

    def verifier(self):
        pass
