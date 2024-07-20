from z3 import *

from tests.debugging.specific_problems.synthesis_problem_test_case_base import SynthesisProblemTestCase


class MaxThreeIntegers(SynthesisProblemTestCase):
    problem_str = """
    (set-logic LIA)
    (synth-fun fn ((vr0 Int) (vr1 Int) (vr2 Int)) Int)
    (declare-var vr0 Int)
    (declare-var vr1 Int)
    (declare-var vr2 Int)
    (constraint (>= (fn vr0 vr1 vr2 ) vr0 ) )
    (constraint (>= (fn vr0 vr1 vr2 ) vr1 ) )
    (constraint (>= (fn vr0 vr1 vr2 ) vr2 ) )
    (constraint (or (= vr0 (fn vr0 vr1 vr2 ) ) ( or (= vr1 ( fn vr0 vr1 vr2 ) ) (= vr2 ( fn vr0 vr1 vr2 )) ) ) )
    (check-synth)
    """

    def test_invalid_and_valid_solutions(self):

        args = [list(self.problem.context.z3_synth_functions.values())[0].domain(i) for i in
                range(list(self.problem.context.z3_synth_functions.values())[0].arity())]
        free_vars = [z3.Var(i, sort) for i, sort in enumerate(args)]

        def incorrect_fn(*args):
            x, y, z = args
            return x + y + z

        incorrect_candidate = incorrect_fn(*free_vars)
        result = self.test_strategy.test_candidates(["incorrect_fn"], [incorrect_candidate])
        self.assertFalse(result)

        def correct_fn(*args):
            x, y, z = args
            return If(x >= y, If(x >= z, x, z), If(y >= z, y, z))

        correct_candidate = correct_fn(*free_vars)
        result = self.test_strategy.test_candidates(["correct_fn"], [correct_candidate])
        self.assertTrue(result)
