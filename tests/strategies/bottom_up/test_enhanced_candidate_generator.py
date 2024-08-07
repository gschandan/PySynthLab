import unittest
import time
from z3 import *
from src.cegis.z3.synthesis_problem_z3 import SynthesisProblemZ3
from src.cegis.z3.candidate_generator.partial_random_candidate_generator import PartialRandomCandidateGenerator
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
        self.generator = PartialRandomCandidateGenerator(self.problem)

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
        self.assertEqual(score, 1.0)

    def test_fuzzy_satisfaction_with_incorrect_candidate(self):
        incorrect_candidate, _ = self.generate_incorrect_candidate()

        self.generator.set_partial_satisfaction_method('fuzzy', True)

        score = self.generator.fuzzy_satisfaction(incorrect_candidate, 'f')

        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)
        self.assertLess(score, 1.0)

    def test_fuzzy_satisfaction_with_partially_correct_candidate(self):
        partially_correct_candidate, _ = self.generate_partially_correct_candidate()

        self.generator.set_partial_satisfaction_method('fuzzy', True)

        score = self.generator.fuzzy_satisfaction(partially_correct_candidate, 'f')

        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_check_partial_satisfaction(self):
        correct_candidate, _ = self.generate_correct_candidate()
        incorrect_candidate, _ = self.generate_incorrect_candidate()
        partially_correct_candidate, _ = self.generate_partially_correct_candidate()

        correct_score = self.generator.check_partial_satisfaction(correct_candidate, 'f')
        incorrect_score = self.generator.check_partial_satisfaction(incorrect_candidate, 'f')
        partial_score = self.generator.check_partial_satisfaction(partially_correct_candidate, 'f')

        self.assertEqual(1.0, correct_score)
        self.assertLess(incorrect_score, 1.0)
        self.assertGreater(partial_score, 0.0)
        self.assertLess(partial_score, 1.0)

    def test_quantitative_satisfaction(self):
        correct_candidate, _ = self.generate_correct_candidate()
        incorrect_candidate, _ = self.generate_incorrect_candidate()
        partially_correct_candidate, _ = self.generate_partially_correct_candidate()

        correct_score = self.generator.quantitative_satisfaction(correct_candidate, 'f')
        incorrect_score = self.generator.quantitative_satisfaction(incorrect_candidate, 'f')
        partial_score = self.generator.quantitative_satisfaction(partially_correct_candidate, 'f')

        self.assertGreater(correct_score, incorrect_score)
        self.assertGreaterEqual(partial_score, incorrect_score)
        self.assertLess(partial_score, correct_score)

    def test_unsat_core_analysis(self):
        correct_candidate, _ = self.generate_correct_candidate()
        incorrect_candidate, _ = self.generate_incorrect_candidate()
        partially_correct_candidate, _ = self.generate_partially_correct_candidate()

        correct_score = self.generator.unsat_core_analysis(correct_candidate, 'f')
        incorrect_score = self.generator.unsat_core_analysis(incorrect_candidate, 'f')
        partial_score = self.generator.unsat_core_analysis(partially_correct_candidate, 'f')
        print(f"correct_score: {correct_score}")
        print(f"incorrect_score: {incorrect_score}")
        print(f"partial_score: {partial_score}")

        self.assertEqual(1.0, correct_score)
        self.assertLess(incorrect_score, 1.0)
        self.assertGreater( partial_score, 0.0)
        self.assertLess(partial_score, 1.0)

    def test_performance(self):
        correct_candidate, _ = self.generate_correct_candidate()

        methods = ['check_partial_satisfaction', 'quantitative_satisfaction', 'unsat_core_analysis', 'fuzzy_satisfaction']

        for method in methods:
            start_time = time.time()
            getattr(self.generator, method)(correct_candidate, 'f')
            end_time = time.time()
            self.assertLess(end_time - start_time, 1.0, f"{method} should complete within 1 second")

    def test_all_satisfaction_methods_with_correct_candidate(self):
        correct_candidate, _ = self.generate_correct_candidate()

        methods = ['check_partial_satisfaction', 'quantitative_satisfaction', 'unsat_core_analysis', 'fuzzy_satisfaction']

        for method in methods:
            score = getattr(self.generator, method)(correct_candidate, 'f')
            self.assertEqual(1.0, score, f"{method}")

    def test_all_satisfaction_methods_with_incorrect_candidate(self):
        incorrect_candidate, _ = self.generate_incorrect_candidate()

        methods = ['check_partial_satisfaction', 'quantitative_satisfaction', 'unsat_core_analysis', 'fuzzy_satisfaction']

        for method in methods:
            score = getattr(self.generator, method)(incorrect_candidate, 'f')
            self.assertLess(score, 1.0, f"{method}")
            self.assertGreaterEqual(score, 0.0, f"{method}")

    def test_all_satisfaction_methods_with_partially_correct_candidate(self):
        partially_correct_candidate, _ = self.generate_partially_correct_candidate()

        methods = ['check_partial_satisfaction', 'quantitative_satisfaction', 'unsat_core_analysis', 'fuzzy_satisfaction']

        for method in methods:
            score = getattr(self.generator, method)(partially_correct_candidate, 'f')
            self.assertLess(score, 1.0, f"{method}")
            self.assertGreater(score, 0.0, f"{method}")


if __name__ == '__main__':
    unittest.main()
