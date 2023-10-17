import os.path
from pysynthlab.synthesis_problem import SynthesisProblem


class ParseSynthesisProblemException(RuntimeError):
    """Unable to parse the provided synthesis problem"""


def sygus_problem_parser(filepath: str = "") -> SynthesisProblem | Exception:
    """
        Parse a SyGuS-IF input file and extract logic type, variables, constraints and functions.
        :param: filepath: The file path to the SyGuS-IF problem file to parse. Defaults to benchmarks folder.
        :type: filepath:str
        :returns: SynthesisProblem: a class containing the parsed problem.
        """
    try:
        if filepath == "" or not os.path.exists(filepath):
            module_directory = os.path.dirname(__file__)
            filepath = os.path.join(module_directory, '..', "benchmarks/lia/small.sl")
        with open(filepath, 'r') as file:
            lines = file.readlines()

        logic = None
        synthesis_function = None
        variables = {}
        constraints = []

        for line in lines:
            line = line.strip()

            if line.startswith('(set-logic'):
                logic = line.split()[1][:-1]

            elif line.startswith('(synth-fun'):
                synthesis_function = line.strip()[:-1]

            elif line.startswith('(declare-var'):
                var = line.split()
                variables[var[1]] = var[2][:-1]

            elif line.startswith('(constraint'):
                constraints.append(line.strip())

    except FileNotFoundError:
        print(f"File '{filepath}' not found.")
        raise ParseSynthesisProblemException(f'File: {filepath} not found')

    return SynthesisProblem(logic, synthesis_function, variables, constraints)
