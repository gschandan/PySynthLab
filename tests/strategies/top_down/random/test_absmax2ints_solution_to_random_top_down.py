import unittest
from typing import Tuple
from unittest.mock import patch
import z3
from src.cegis.z3.synthesis_problem import SynthesisProblem, Options
from src.cegis.z3.synthesis_strategy.random_search_top_down import RandomSearchStrategyTopDown


class TestValidCandidateDirectlyForAbsMax2Ints(unittest.TestCase):
    def setUp(self):
        self.problem_str = """
        (set-logic LIA)
        (synth-fun f ((a Int) (b Int)) Int)
        (declare-var x Int)
        (declare-var y Int)
        (constraint (= (f x y) (f y x)))
        (constraint (and (<= x (f x y)) (<= y (f x y))))
        """
        self.options = Options()
        self.problem = SynthesisProblem(self.problem_str, self.options)
        self.strategy = RandomSearchStrategyTopDown(self.problem)

    def generate_correct_abs_max_function(self) -> Tuple[z3.ExprRef, str]:
        vars = [z3.Var(i, z3.IntSort()) for i in range(2)]
        
        expr = z3.If(z3.If(vars[0] >= 0, vars[0], -vars[0]) > z3.If(vars[1] >= 0, vars[1], -vars[1]),
                     z3.If(vars[0] >= 0, vars[0], -vars[0]),
                     z3.If(vars[1] >= 0, vars[1], -vars[1]))

        func_str = f"def absolute_max_function(x, y):\n"
        func_str += f"    return {expr}\n"

        return expr, func_str

    @patch.object(RandomSearchStrategyTopDown, 'generate_random_term')
    def test_strategy_finds_correct_function(self, mock_generate):
        correct_func, correct_func_str = self.generate_correct_abs_max_function()
        mock_generate.return_value = (correct_func, correct_func_str)

        self.strategy.execute_cegis()

        self.assertIsNotNone(self.problem.context.z3_synth_functions['f'],"Should find a satisfying function")

        self.assertTrue(self.strategy.solution_found, "Should have found a solution")




if __name__ == '__main__':
    unittest.main()
