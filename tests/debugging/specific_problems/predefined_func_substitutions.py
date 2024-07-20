from z3 import *
from tests.debugging.specific_problems.synthesis_problem_test_case_base import SynthesisProblemTestCase


class SubstitutionCheck(SynthesisProblemTestCase):

    problem_str = """
        (set-logic LIA)
        (define-fun double ((x Int)) Int (* x 2))
        (define-fun inc ((x Int)) Int (+ x 1))
        (synth-fun f ((x Int)) Int
            ((Start Int (x
                         0
                         1
                         (double Start)
                         (inc Start)
                         (+ Start Start)
                         (- Start Start)))))
        (declare-var x Int)
        (constraint (= (f x) (double (inc x))))
        (check-synth)
        """

    def test_invalid_and_valid_solutions(self):
        x = Int('x')

        def incorrect_f(x):
            return x + 1

        incorrect_candidates = [incorrect_f(x)]
        incorrect_func_strs = ["incorrect_f"]

        result = self.test_strategy.test_candidates(incorrect_func_strs, incorrect_candidates)
        self.assertFalse(result)

        def correct_f(x):
            return 2 * (x + 1)

        correct_candidates = [correct_f(x)]
        correct_func_strs = ["correct_f"]

        result = self.test_strategy.test_candidates(correct_func_strs, correct_candidates)
        self.assertTrue(result)