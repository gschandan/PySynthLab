from z3 import *
from tests.debugging.specific_problems.synthesis_problem_test_case_base import SynthesisProblemTestCase


class LoopProblem(SynthesisProblemTestCase):
    problem_str = """
        (set-logic LIA)
        (define-fun qm ((a Int) (b Int)) Int (ite (< a 0) b a))
        (synth-fun qm-loop ((x Int)) Int
        ((Start Int (x 0 1 3
        (- Start Start)
        (+ Start Start)
        (qm Start Start)))))
        (declare-var x Int)
        (constraint (= (qm-loop x) (ite (<= x 0) 3 (- x 1))))
        (check-synth)
        """

    def test_invalid_and_valid_solutions(self):
        x = Int('x')

        def incorrect_qm_loop(x):
            return x + 1

        incorrect_candidates = [incorrect_qm_loop(x)]
        incorrect_func_strs = ["incorrect_qm_loop"]

        result = self.test_strategy.test_candidates(incorrect_func_strs, incorrect_candidates)
        self.assertFalse(result)

        def correct_qm_loop(x):
            return If(x <= 0, 3, x - 1)

        correct_candidates = [correct_qm_loop(x)]
        correct_func_strs = ["correct_qm_loop"]

        result = self.test_strategy.test_candidates(correct_func_strs, correct_candidates)
        self.assertTrue(result)