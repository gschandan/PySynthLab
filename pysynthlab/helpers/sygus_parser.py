import os.path

from pysynthlab import synthesis_problem
from pysynthlab.synthesis_problem import SynthesisProblem


class ParseSynthesisProblemException(RuntimeError):
    """Unable to parse the provided synthesis problem"""


def sygus_problem_parser(filename: str, filepath: str = "") -> SynthesisProblem | Exception:
    """
        Parse a SyGuS-IF input file and extract logic type, variables, constraints and functions.
        :param: filename: The filename of the SyGuS-IF problem file to parse.
        :type: filename:str
        :param: filepath: The file path to the SyGuS-IF problem file to parse. Defaults to benchmarks folder.
        :type: filepath:str
        :returns: SynthesisProblem: a class containing the parsed problem.
        """
    try:
        if filepath == "" or not os.path.exists(filepath):
            module_directory = os.path.dirname(__file__)
            filepath = os.path.join(module_directory, "benchmarks/lia")
        with open(os.path.join(filepath, filename), 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        raise ParseSynthesisProblemException(f'File: {filename} not found at {filepath}')

    return SynthesisProblem("something")
