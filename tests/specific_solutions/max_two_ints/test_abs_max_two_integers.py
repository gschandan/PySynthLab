import unittest
from typing import List, Tuple, Callable
from z3 import *
from src.cegis.z3.synthesis_strategy.random_search_bottom_up import  SynthesisProblem
from src.cegis.z3.synthesis_problem import Options

class GivenTheMaxOfTwoIntegersProblem(unittest.TestCase):
    def setUp(self):
        self.problem_str = """
        (synth-fun f ((x Int) (y Int)) Int)
        (declare-var x Int)
        (declare-var y Int)
        (constraint (= (f x y) (f y x)))
        (constraint (and (<= x (f x y)) (<= y (f x y))))
        """
        self.options = Options()
        self.problem = SynthesisProblem(self.problem_str, self.options)

    def test_valid_solutions_Are_correctly_identified(self):
        def generate_correct_abs_max_function(arg_sorts: List[z3.SortRef]) -> Tuple[Callable, str]:
            args = [z3.Var(i, sort) for i, sort in enumerate(arg_sorts)]

            def absolute_max_function(*values):
                if len(values) != 2:
                    raise ValueError("absolute_max_function expects exactly 2 arguments.")
                x, y = values
                return If(If(x >= 0, x, -x) > If(y >= 0, y, -y), If(x >= 0, x, -x), If(y >= 0, y, -y))
    
            expr = absolute_max_function(*args[:2])
            func_str = f"def absolute_max_function({', '.join(str(arg) for arg in args[:2])}):\n"
            func_str += f"    return {str(expr)}\n"
            return absolute_max_function, func_str

        SynthesisProblem.logger.info("Trying known candidate for max")
        args = [list(self.problem.context.z3_synth_functions.values())[0].domain(i) for i in
                range(list(self.problem.context.z3_synth_functions.values())[0].arity())]
        candidate, func_str = generate_correct_abs_max_function(args)

        free_variables = [z3.Var(i, sort) for i, sort in enumerate(args)]
        candidate_function = candidate(*free_variables)

        SynthesisProblem.logger.info(f"candidate_function for substitution {candidate_function}")
        SynthesisProblem.logger.info(f"Testing guess: {func_str}")
        result = self.problem.test_candidates_alternative([func_str],
                                              [candidate_function])
        SynthesisProblem.logger.info("\n")
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
