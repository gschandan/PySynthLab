import unittest
from typing import List, Tuple
from unittest.mock import patch
import z3
from src.cegis.z3.synthesis_problem import SynthesisProblem, SynthesisProblemOptions
from src.cegis.z3.random_search_top_down import RandomSearchStrategyTopDown


class TestValidCandidateDirectlyForAbsMax2Ints(unittest.TestCase):
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
        vars = [z3.Var(i, z3.IntSort()) for i in range(2)]
        
        expr = z3.If(z3.If(vars[0] >= 0, vars[0], -vars[0]) > z3.If(vars[1] >= 0, vars[1], -vars[1]),
                     z3.If(vars[0] >= 0, vars[0], -vars[0]),
                     z3.If(vars[1] >= 0, vars[1], -vars[1]))


        return expr, 'f'

    @patch.object(RandomSearchStrategyTopDown, 'generate_random_term')
    def test_strategy_finds_correct_function(self, mock_generate):
        correct_func, correct_func_str = self.generate_correct_abs_max_function()
        mock_generate.return_value = (correct_func, correct_func_str)

        self.strategy.execute_cegis()

        self.assertIsNotNone(self.problem.context.z3_synth_functions['f'],"Should find a satisfying function")

        self.assertTrue(self.strategy.solution_found, "Should have found a solution")




if __name__ == '__main__':
    unittest.main()
