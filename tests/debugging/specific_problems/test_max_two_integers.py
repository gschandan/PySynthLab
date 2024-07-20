from z3 import *
from tests.debugging.specific_problems.synthesis_problem_test_case_base import SynthesisProblemTestCase


class MaxTwoIntegers(SynthesisProblemTestCase):
    problem_str = """
    (set-logic LIA)
    (synth-fun max2 ((a Int) (b Int)) Int)
    (declare-var x Int)
    (declare-var y Int)
    (constraint (>= (max2 x y) x))
    (constraint (>= (max2 x y) y))
    (constraint (or (= x (max2 x y)) (= y (max2 x y))))
    (constraint (= (max2 x x) x))
    (constraint (forall ((x Int) (y Int))
    (=> (>= x y) (= (max2 x y) x))))
    (constraint (forall ((x Int) (y Int))
    (=> (>= y x) (= (max2 y x) y))))
    (check-synth)
    """

    def test_invalid_and_valid_solutions(self):

        args = [list(self.problem.context.z3_synth_functions.values())[0].domain(i) for i in
                range(list(self.problem.context.z3_synth_functions.values())[0].arity())]
        free_vars = [z3.Var(i, sort) for i, sort in enumerate(args)]

        def incorrect_fn(*args):
            x, y = args
            return x + y

        incorrect_candidate = incorrect_fn(*free_vars)
        result = self.test_strategy.test_candidates(["incorrect_fn"], [incorrect_candidate])
        self.assertFalse(result)

        def correct_fn(*args):
            x, y = args
            return If(x <= y, y, x)

        correct_candidate = correct_fn(*free_vars)
        result = self.test_strategy.test_candidates(["correct_fn"], [correct_candidate])
        self.assertTrue(result)
