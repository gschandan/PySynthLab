import unittest
from z3 import *

from src.cegis.z3.random_search_bottom_up import SynthesisProblem
from src.cegis.z3.synthesis_problem import SynthesisProblemOptions


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

    def test_invalid_and_valid_solutions(self):
        x = z3.Var(0, IntSort())

        def incorrect_id1(*values):
            x = values[0] + 1
            return x

        def incorrect_id2(*values):
            return values[0] - 1

        def incorrect_id3(*values):
            return z3.IntVal(0)

        def incorrect_id4(*values):
            return -values[0]

        incorrect_candidates = [incorrect_id1(x), incorrect_id2(x), incorrect_id3(x), incorrect_id4(x)]
        incorrect_func_strs = ["incorrect_id1", "incorrect_id2", "incorrect_id3", "incorrect_id4"]

        self.problem.print_msg(f"Testing incorrect candidates: {'; '.join(incorrect_func_strs)}", level=1)
        result = self.problem.test_candidates(incorrect_func_strs, incorrect_candidates)
        self.problem.print_msg("\n", level=1)
        self.assertFalse(result)

        def correct_id1(*values):
            return values[0]

        partially_correct_candidates = [correct_id1(x), incorrect_id2(x), incorrect_id3(x), incorrect_id4(x)]
        partially_correct_func_strs = ["correct_id1", "incorrect_id2", "incorrect_id3", "incorrect_id4"]

        self.problem.print_msg(f"Testing partially correct candidates: {'; '.join(partially_correct_func_strs)}",
                               level=1)
        result = self.problem.test_candidates(partially_correct_func_strs, partially_correct_candidates)
        self.problem.print_msg("\n", level=1)
        self.assertFalse(result)

        def correct_id2(*values):
            return values[0]

        def correct_id3(*values):
            return values[0]

        def correct_id4(*values):
            return values[0]

        correct_candidates = [correct_id1(x), correct_id2(x), correct_id3(x), correct_id4(x)]
        correct_func_strs = ["correct_id1", "correct_id2", "correct_id3", "correct_id4"]

        self.problem.print_msg(f"Testing correct candidates: {'; '.join(correct_func_strs)}", level=1)
        result = self.problem.test_candidates(correct_func_strs, correct_candidates)
        self.problem.print_msg("\n", level=1)
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
