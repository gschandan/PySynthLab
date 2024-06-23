import unittest
from unittest.mock import patch
from typing import List, Tuple
import z3
from src.cegis.z3.synthesis_problem import SynthesisProblem, SynthesisProblemOptions
from src.cegis.z3.random_search_top_down import RandomSearchStrategyTopDown


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
        self.problem = SynthesisProblem(self.problem_str, self.options)
        self.strategy = RandomSearchStrategyTopDown(self.problem)

    def generate_correct_abs_max_function(self) -> Tuple[z3.ExprRef, str]:
        x, y = z3.Ints('x y')
        expr = z3.If(z3.If(x >= 0, x, -x) > z3.If(y >= 0, y, -y),
                     z3.If(x >= 0, x, -x),
                     z3.If(y >= 0, y, -y))
        func_str = f"def absolute_max_function(x, y):\n    return {expr}\n"
        return expr, func_str

    def generate_incorrect_functions(self) -> List[Tuple[z3.ExprRef, str]]:
        x, y = z3.Ints('x y')
        return [
            (x + y, "def f1(x, y):\n    return x + y\n"),
            (x - y, "def f2(x, y):\n    return x - y\n"),
            (x * y, "def f3(x, y):\n    return x * y\n"),
            (z3.If(x < y, x, y), "def f4(x, y):\n    return x if x < y else y\n")
        ]

    @patch.object(RandomSearchStrategyTopDown, 'generate_arithmetic_function')
    def test_strategy_handles_counterexamples_and_finds_correct_function(self, mock_generate):
        incorrect_functions = self.generate_incorrect_functions()
        correct_func, correct_func_str = self.generate_correct_abs_max_function()

        mock_generate.side_effect = incorrect_functions + [(correct_func, correct_func_str)]

        self.strategy.execute_cegis()

        self.assertEqual(mock_generate.call_count, len(incorrect_functions) + 1,"Should try all incorrect functions before the correct one")
        self.assertIsNotNone(self.problem.context.z3_synth_functions['f'],"Should find a satisfying function")
        self.assertTrue(self.strategy.get_solution_found(), "Should have found a solution")

        self.assertEqual(len(self.problem.context.counterexamples), len(incorrect_functions),"Counterexamples should have been generated for incorrect functions")

        print("\nCounterexamples generated:")
        for i, (func_str, ce, _) in enumerate(self.problem.context.counterexamples):
            print(f"Function {i + 1}: {func_str}")
            print(f"Counterexample: {ce}")


if __name__ == '__main__':
    unittest.main()
