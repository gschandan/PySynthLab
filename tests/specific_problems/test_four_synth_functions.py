import unittest
from z3 import *

from src.cegis.z3.synthesis_problem_z3 import SynthesisProblemOptions, SynthesisProblem


class FourSynthFunctions(unittest.TestCase):
    def setUp(self):
        self.problem_str = """
            (set-logic LIA)
            (synth-fun id1 ((x Int)) Int (
                (Start Int (x (- Start) (+ Start x)))
            ))
            (synth-fun id2 ((x Int)) Int (
                (Start Int (x (- Start) (+ Start x)))
            ))
            (synth-fun id3 ((x Int)) Int (
                (Start Int (0 (- Start) (+ Start x)))
            ))
            (synth-fun id4 ((x Int)) Int (
                (Start Int (x (- Start) (+ Start x)))
            ))
            (declare-var x Int)
            (constraint (= (id1 x) (id2 x) (id3 x) (id4 x) x))
            (check-synth)
        """
        self.options = SynthesisProblemOptions()
        self.problem = SynthesisProblem(self.problem_str, self.options)

    def test_valid_solutions_are_correctly_identified(self):
        def id1_function(*values):
            x = values[0]
            return x

        def id2_function(*values):
            x = values[0]
            return x

        def id3_function(*values):
            x = values[0]
            return x

        def id4_function(*values):
            x = values[0]
            return x

        candidate_functions = [id1_function, id2_function, id3_function, id4_function]
        func_strs = ["id1_function", "id2_function", "id3_function", "id4_function"]
        args_list = [z3.Var(0, IntSort())]
        candidate_functions = [f(*args_list) for f in candidate_functions]
        self.problem.print_msg(f"candidate_functions for substitution {candidate_functions}", level=0)
        self.problem.print_msg(f"Testing known candidate: {'; '.join(func_strs)}", level=1)
        result = self.problem.test_multiple_candidates( func_strs,
                                                       candidate_functions)
        self.problem.print_msg("\n", level=1)
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
