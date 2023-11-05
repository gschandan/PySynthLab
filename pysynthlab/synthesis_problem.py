from helpers.parser.src.v2.parser import SygusV2Parser
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
    sygus_version :int = 0

    def __init__(self, problem: str):
        """
        Constructs all the necessary attributes for the synthesis problem.

        Parameters
        ----------
            problem : str
                synthesis problem to be parsed
        """
        self.parser = SygusV2Parser()
        self.problem = self.parser.parse(problem)
        self.printer = SygusV2ASTPrinter()


    def __str__(self) -> str:
        """
        Returns a string representation of the synthesis problem.
        :returns: str: A string representation of the synthesis problem.
        """
        output: list[str] = [f"(set-logic {self.logic})\n", self.synthesis_function + ')\n']

        for var_name, var_type in self.variables.items():
            output.append(f"(declare-var {var_name} {var_type})")

        output.extend(self.constraints)

        return '\n'.join(output)

    def info(self) -> None:
        """ Prints the synthesis problem to the console
        :returns: None
        """
        print(self)
