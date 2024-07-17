import unittest
from unittest.mock import patch
from typing import List, Tuple
import z3

from src.cegis.z3.synthesis_problem import SynthesisProblem, SynthesisProblemOptions
from src.cegis.z3.synthesis_strategy.random_search_top_down import RandomSearchStrategyTopDown


class TestValidCandidateAfterSeveralInvalidCandidatesAndCounterexamples(unittest.TestCase):
    def setUp(self):
        self.problem_str = """
        (set-logic LIA)
        (synth-fun f ((x Int) (y Int)) Int)
        (declare-var x Int)
        (declare-var y Int)
        (constraint (= (f x y) (f y x)))
        (constraint (and (<= x (f x y)) (<= y (f x y))))
        """
        self.options = SynthesisProblemOptions()
        self.options.max_candidates_at_each_depth = 10
        self.problem = SynthesisProblem(self.problem_str, self.options)
        self.strategy = RandomSearchStrategyTopDown(self.problem)

    def generate_correct_max_function(self) -> Tuple[z3.ExprRef, str]:
        vars = [z3.Var(i, z3.IntSort()) for i in range(2)]
        return z3.If(vars[0] >= vars[1], vars[0], vars[1]), 'f'

    def generate_incorrect_functions(self) -> List[Tuple[z3.ExprRef, str]]:
        vars = [z3.Var(i, z3.IntSort()) for i in range(2)]
        return [
            (vars[0] + vars[1], 'f'),
            (vars[0] - vars[1], 'f'),
            (vars[0] * 2, 'f'),
            (z3.If(vars[0] < vars[1], vars[0], vars[1]), 'f')
        ]

    @patch.object(RandomSearchStrategyTopDown, 'generate_candidates')
    def test_strategy_handles_counterexamples_and_finds_correct_function(self, mock_generate):
        incorrect_functions = self.generate_incorrect_functions()
        correct_func, correct_func_name = self.generate_correct_max_function()

        mock_data = [incorrect_functions[i % len(incorrect_functions)] for i in
                     range(self.options.max_candidates_at_each_depth - 1)]
        mock_data.append((correct_func, correct_func_name))

        mock_generate.side_effect = [[func] for func in mock_data]

        self.strategy.execute_cegis()
        self.assertEqual(mock_generate.call_count, self.options.max_candidates_at_each_depth,
                         f"Should try {self.options.max_candidates_at_each_depth} times before finding the correct one")
        self.assertIsNotNone(self.problem.context.z3_synth_functions['f'],
                             "Should find a satisfying function")
        self.assertTrue(self.strategy.solution_found, "Should have found a solution")

        print("\nCounterexamples generated:")
        for i, (func_name, ce, _) in enumerate(self.problem.context.counterexamples):
            print(f"Function {i + 1}: {func_name}")
            print(f"Counterexample: {ce}")


if __name__ == '__main__':
    unittest.main()
