import unittest
import time
from z3 import *
from src.cegis.z3.synthesis_problem_z3 import SynthesisProblemZ3
from src.cegis.z3.candidate_generator.enhanced_random_candidate_generator import EnhancedRandomCandidateGenerator
from src.utilities.options import Options


class TestEnhancedRandomCandidateGenerator(unittest.TestCase):
    def setUp(self):
        self.problem_str = """
        (set-logic LIA)
        (synth-fun f ((x Int) (y Int)) Int)

        (declare-var x Int)
        (declare-var y Int)

        (constraint (=> (> x y) (= (f x y) x)))
        (constraint (=> (<= x y) (= (f x y) y)))

        (check-synth)
        """
        self.options = Options()
        self.problem = SynthesisProblemZ3(self.problem_str, self.options)
        self.generator = EnhancedRandomCandidateGenerator(self.problem, self.metrics)

    def generate_correct_candidate(self) -> tuple[ExprRef, str]:
        vars = [Var(i, IntSort()) for i in range(2)]

        expr = If(vars[0] > vars[1], vars[0], vars[1])

        func_str = "def correct_function(x, y):\n"
        func_str += "    return x if x > y else y\n"

        return expr, func_str

    def generate_incorrect_candidate(self) -> tuple[ExprRef, str]:
        vars = [Var(i, IntSort()) for i in range(2)]

        expr = vars[0]

        func_str = "def incorrect_function(x, y):\n"
        func_str += "    return x\n"

        return expr, func_str

    def generate_partially_correct_candidate(self) -> tuple[ExprRef, str]:
        vars = [Var(i, IntSort()) for i in range(2)]

        expr = If(vars[0] > vars[1], vars[0], vars[0])

        func_str = "def partially_correct_function(x, y):\n"
        func_str += "    return x if x > y else x\n"

        return expr, func_str

    def test_fuzzy_satisfaction_with_implies(self):
        correct_candidate, _ = self.generate_correct_candidate()

        self.generator.set_partial_satisfaction_method('fuzzy', True)

        score = self.generator.fuzzy_satisfaction(correct_candidate, 'f')

        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)
        self.assertEqual(score, 1.0, "The correct candidate should fully satisfy the constraints")

    def test_fuzzy_satisfaction_with_incorrect_candidate(self):
        incorrect_candidate, _ = self.generate_incorrect_candidate()

        self.generator.set_partial_satisfaction_method('fuzzy', True)

        score = self.generator.fuzzy_satisfaction(incorrect_candidate, 'f')

        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)
        self.assertLess(score, 1.0, "The incorrect candidate should not fully satisfy the constraints")

    def test_fuzzy_satisfaction_with_partially_correct_candidate(self):
        partially_correct_candidate, _ = self.generate_partially_correct_candidate()

        self.generator.set_partial_satisfaction_method('fuzzy', True)

        score = self.generator.fuzzy_satisfaction(partially_correct_candidate, 'f')

        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)
        self.assertGreater(score, 0.0, "The partially correct candidate should satisfy some constraints")
        self.assertLess(score, 1.0, "The partially correct candidate should not fully satisfy all constraints")

    def test_check_partial_satisfaction(self):
        correct_candidate, _ = self.generate_correct_candidate()
        incorrect_candidate, _ = self.generate_incorrect_candidate()
        partially_correct_candidate, _ = self.generate_partially_correct_candidate()

        correct_score = self.generator.check_partial_satisfaction(correct_candidate, 'f')
        incorrect_score = self.generator.check_partial_satisfaction(incorrect_candidate, 'f')
        partial_score = self.generator.check_partial_satisfaction(partially_correct_candidate, 'f')

        self.assertEqual(correct_score, 1.0, "The correct candidate should fully satisfy the constraints")
        self.assertLess(incorrect_score, 1.0, "The incorrect candidate should not fully satisfy the constraints")
        self.assertGreater(partial_score, 0.0, "The partially correct candidate should satisfy some constraints")
        self.assertLess(partial_score, 1.0, "The partially correct candidate should not fully satisfy all constraints")

    def test_check_with_soft_constraints(self):
        correct_candidate, _ = self.generate_correct_candidate()
        incorrect_candidate, _ = self.generate_incorrect_candidate()
        partially_correct_candidate, _ = self.generate_partially_correct_candidate()

        correct_score = self.generator.check_with_soft_constraints(correct_candidate, 'f')
        incorrect_score = self.generator.check_with_soft_constraints(incorrect_candidate, 'f')
        partial_score = self.generator.check_with_soft_constraints(partially_correct_candidate, 'f')

        self.assertEqual(correct_score, 1.0, "The correct candidate should fully satisfy the constraints")
        self.assertLess(incorrect_score, 1.0, "The incorrect candidate should not fully satisfy the constraints")
        self.assertGreater(partial_score, 0.0, "The partially correct candidate should satisfy some constraints")
        self.assertLess(partial_score, 1.0, "The partially correct candidate should not fully satisfy all constraints")

    def test_max_smt_satisfaction(self):
        correct_candidate, _ = self.generate_correct_candidate()
        incorrect_candidate, _ = self.generate_incorrect_candidate()
        partially_correct_candidate, _ = self.generate_partially_correct_candidate()

        correct_score = self.generator.max_smt_satisfaction(correct_candidate, 'f')
        incorrect_score = self.generator.max_smt_satisfaction(incorrect_candidate, 'f')
        partial_score = self.generator.max_smt_satisfaction(partially_correct_candidate, 'f')

        self.assertEqual(correct_score, 1.0, "The correct candidate should fully satisfy the constraints")
        self.assertLess(incorrect_score, 1.0, "The incorrect candidate should not fully satisfy the constraints")
        self.assertGreater(partial_score, 0.0, "The partially correct candidate should satisfy some constraints")
        self.assertLess(partial_score, 1.0, "The partially correct candidate should not fully satisfy all constraints")

    def test_quantitative_satisfaction(self):
        correct_candidate, _ = self.generate_correct_candidate()
        incorrect_candidate, _ = self.generate_incorrect_candidate()
        partially_correct_candidate, _ = self.generate_partially_correct_candidate()

        correct_score = self.generator.quantitative_satisfaction(correct_candidate, 'f')
        incorrect_score = self.generator.quantitative_satisfaction(incorrect_candidate, 'f')
        partial_score = self.generator.quantitative_satisfaction(partially_correct_candidate, 'f')

        self.assertGreater(correct_score, incorrect_score, "The correct candidate should have a higher score than the incorrect one")
        self.assertGreater(partial_score, incorrect_score, "The partially correct candidate should have a higher score than the incorrect one")
        self.assertLess(partial_score, correct_score, "The partially correct candidate should have a lower score than the correct one")

    def test_unsat_core_analysis(self):
        correct_candidate, _ = self.generate_correct_candidate()
        incorrect_candidate, _ = self.generate_incorrect_candidate()
        partially_correct_candidate, _ = self.generate_partially_correct_candidate()

        correct_score = self.generator.unsat_core_analysis(correct_candidate, 'f')
        incorrect_score = self.generator.unsat_core_analysis(incorrect_candidate, 'f')
        partial_score = self.generator.unsat_core_analysis(partially_correct_candidate, 'f')

        self.assertEqual(correct_score, 1.0, "The correct candidate should fully satisfy the constraints")
        self.assertLess(incorrect_score, 1.0, "The incorrect candidate should not fully satisfy the constraints")
        self.assertGreater(partial_score, 0.0, "The partially correct candidate should satisfy some constraints")
        self.assertLess(partial_score, 1.0, "The partially correct candidate should not fully satisfy all constraints")

    def test_performance(self):
        correct_candidate, _ = self.generate_correct_candidate()

        methods = ['check_partial_satisfaction', 'check_with_soft_constraints', 'max_smt_satisfaction',
                   'quantitative_satisfaction', 'unsat_core_analysis', 'fuzzy_satisfaction']

        for method in methods:
            start_time = time.time()
            getattr(self.generator, method)(correct_candidate, 'f')
            end_time = time.time()
            self.assertLess(end_time - start_time, 1.0, f"{method} should complete within 1 second")


if __name__ == '__main__':
    unittest.main()