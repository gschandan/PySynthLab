import time
import unittest

from z3 import *

from src.cegis.z3.synthesis_strategy.cegis_t import CegisT, CandidateType
from src.utilities.options import Options
from src.cegis.z3.synthesis_problem import SynthesisProblem

@unittest.skip("need to fix")
class TestCegisT(unittest.TestCase):
    def setUp(self):
        self.problem_str = """
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
        self.options = Options()
        self.problem = SynthesisProblem(self.problem_str, self.options)
        self.strategy = CegisT(self.problem)

    def test_verify_correct_candidate(self):
        def max2_correct(x, y):
            return If(x >= y, x, y)

        candidate: CandidateType = {
            'max2': ([], max2_correct(Var(0, IntSort()), Var(1, IntSort())))
        }

        result = self.strategy.verify(candidate)
        self.assertIsNone(result, "Correct candidate should not produce a counterexample")

    def test_verify_correct_candidate_manual(self):
        def max2_correct(x, y):
            return If(x >= y, x, y)

        candidate: CandidateType = {
            'max2': ([], max2_correct(Var(0, IntSort()), Var(1, IntSort())))
        }

        solver = self.problem.context.verification_solver
        solver.push()

        substitutions = []
        for func_name, (entries, else_value) in candidate.items():
            func = self.problem.context.z3_synth_functions[func_name]
            args = [Var(i, func.domain(i)) for i in range(func.arity())]
            body = else_value
            for entry_args, entry_value in reversed(entries):
                condition = And(*[arg == entry_arg for arg, entry_arg in zip(args, entry_args)])
                body = If(condition, entry_value, body)

            def func_appl(*values):
                return body

            substitutions.append((func, func_appl(*args)))

        substituted_neg_constraints = []
        constraints = self.problem.context.z3_negated_constraints
        for constraint in constraints:
            substituted_neg_constraints.append(substitute_funs(constraint, substitutions))

        result = self.strategy.verify(candidate)
        self.assertIsNone(result, "Correct candidate should not produce a counterexample")

    def test_verify_incorrect_candidate(self):
        def max2_incorrect(x, y):
            return x

        candidate: CandidateType = {
            'max2': ([], max2_incorrect(Var(0, IntSort()), Var(1, IntSort())))
        }

        result = self.strategy.verify(candidate)
        self.assertIsNotNone(result, "Incorrect candidate should produce a counterexample")
        self.assertIsInstance(result, dict, "Counterexample should be a dictionary")
        self.assertIn('x', result, "Counterexample should contain 'x'")
        self.assertIn('y', result, "Counterexample should contain 'y'")

    def test_verify_with_multiple_functions(self):
        problem_str_multi = """
        (set-logic LIA)
        (synth-fun max2 ((a Int) (b Int)) Int)
        (synth-fun min2 ((a Int) (b Int)) Int)

        (declare-var x Int)
        (declare-var y Int)

        (constraint (>= (max2 x y) (min2 x y)))
        (constraint (or (= x (max2 x y)) (= y (max2 x y))))
        (constraint (or (= x (min2 x y)) (= y (min2 x y))))
        (check-synth)
        """
        problem_multi = SynthesisProblem(problem_str_multi, self.options)
        strategy_multi = CegisT(problem_multi)

        def max2_correct(x, y):
            return If(x >= y, x, y)

        def min2_correct(x, y):
            return If(x <= y, x, y)

        candidate: CandidateType = {
            'max2': ([], max2_correct(Var(0, IntSort()), Var(1, IntSort()))),
            'min2': ([], min2_correct(Var(0, IntSort()), Var(1, IntSort())))
        }

        result = strategy_multi.verify(candidate)
        self.assertIsNone(result, "Correct candidate for multiple functions should not produce a counterexample")

    def test_verify_performance(self):
        def max2_correct(x, y):
            return If(x >= y, x, y)

        candidate: CandidateType = {
            'max2': ([], max2_correct(Var(0, IntSort()), Var(1, IntSort())))
        }

        start_time = time.time()
        self.strategy.verify(candidate)
        end_time = time.time()

        self.assertLess(end_time - start_time, 1.0, "Verify method should complete within 1 second")


if __name__ == '__main__':
    unittest.main()
