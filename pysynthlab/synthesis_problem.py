from typing import List


class SynthesisProblem:
    """
        A class representing a synthesis problem.
        ...
        Attributes
        ----------
        logic : str
            logic for the synthesis problem e.g. LIA (Linear Integer Arithmetic)
        synthesis_function : str
            the function to be synthesised
        variables : dict
            variables to be used: <name,type>
        constraints : list
            constraints to specify a desired property of the synthesised function
        Methods
        -------
        info():
            Prints the synthesis problem to the console
        """

    def __init__(self, logic: str, synthesis_function: str, variables: dict[str, str], constraints: list[str]):
        """
        Constructs all the necessary attributes for the synthesis problem.

        Parameters
        ----------
            logic : str
                logic for the synthesis problem e.g. LIA (Linear Integer Arithmetic)
            synthesis_function : str
                the function to be synthesised
            variables : dict
                variables to be used: <name,type>
            constraints : list
                constraints to specify a desired property of the synthesised function
        """
        self.logic: str = logic
        self.synthesis_function: str = synthesis_function
        self.variables: dict[str, str] = variables
        self.constraints: list[str] = constraints

    def __str__(self):
        """
        Returns a string representation of the synthesis problem.

        Returns:
            str: A string representation of the synthesis problem.
        """
        output: list[str] = [f"(set-logic {self.logic})\n", self.synthesis_function, '\n']

        for var_name, var_type in self.variables.items():
            output.append(f"(declare-var {var_name} {var_type})")

        output.extend(self.constraints)

        return '\n'.join(output)

    def info(self) -> None:
        """ Prints the synthesis problem to the console
        :returns: None
        """
        print(self)

