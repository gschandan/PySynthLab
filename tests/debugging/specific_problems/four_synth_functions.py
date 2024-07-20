from z3 import *
from tests.debugging.specific_problems.synthesis_problem_test_case_base import SynthesisProblemTestCase


class FourSynthFunctions(SynthesisProblemTestCase):
    problem_str = """
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

    def test_invalid_and_valid_solutions(self):
        x = Int('x')

        def incorrect_id1(x):
            return x + 1

        def incorrect_id2(x):
            return x - 1

        def incorrect_id3(x):
            return IntVal(0)

        def incorrect_id4(x):
            return -x

        incorrect_candidates = [incorrect_id1(x), incorrect_id2(x), incorrect_id3(x), incorrect_id4(x)]
        incorrect_func_strs = ["incorrect_id1", "incorrect_id2", "incorrect_id3", "incorrect_id4"]

        result = self.test_strategy.test_candidates(incorrect_func_strs, incorrect_candidates)
        self.assertFalse(result)

        def correct_id1(x):
            return x

        partially_correct_candidates = [correct_id1(x), incorrect_id2(x), incorrect_id3(x), incorrect_id4(x)]
        partially_correct_func_strs = ["correct_id1", "incorrect_id2", "incorrect_id3", "incorrect_id4"]

        result = self.test_strategy.test_candidates(partially_correct_func_strs, partially_correct_candidates)
        self.assertFalse(result)

        def correct_id2(x):
            return x

        def correct_id3(x):
            return x

        def correct_id4(x):
            return x

        correct_candidates = [correct_id1(x), correct_id2(x), correct_id3(x), correct_id4(x)]
        correct_func_strs = ["correct_id1", "correct_id2", "correct_id3", "correct_id4"]

        result = self.test_strategy.test_candidates(correct_func_strs, correct_candidates)
        self.assertTrue(result)
