import unittest
from typing import List, Tuple, Callable
from z3 import *
from src.cegis.z3.random_search_bottom_up import  SynthesisProblem
from src.cegis.z3.synthesis_problem import SynthesisProblemOptions


class GivenTheMaxOfTwoIntegersProblem(unittest.TestCase):
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
        self.options = SynthesisProblemOptions()
        self.problem = SynthesisProblem(self.problem_str, self.options)

    def test_valid_solutions_Are_correctly_identified(self):
        def generate_max_function( arg_sorts: List[z3.SortRef]) -> Tuple[Callable, str]:
            args = [z3.Var(i, sort) for i, sort in enumerate(arg_sorts)]

            def max_function(*values):
                if len(values) != 2:
                    raise ValueError("max_function expects exactly 2 arguments.")
                x, y = values
                return If(x <= y, y, x)

            expr = max_function(*args[:2])
            func_str = f"def max_function({', '.join(str(arg) for arg in args[:2])}):\n"
            func_str += f"    return {str(expr)}\n"
            return max_function, func_str

        self.problem.print_msg("Trying known candidate for max", level=0)
        args = [list(self.problem.context.z3_synth_functions.values())[0].domain(i) for i in
                range(list(self.problem.context.z3_synth_functions.values())[0].arity())]
        candidate, func_str = generate_max_function(args)

        free_variables = [z3.Var(i, sort) for i, sort in enumerate(args)]
        candidate_function = candidate(*free_variables)

        self.problem.print_msg(f"candidate_function for substitution {candidate_function}", level=0)
        self.problem.print_msg(f"Testing guess: {func_str}", level=1)
        result = self.problem.test_candidates([func_str],
                                              [candidate_function])
        self.problem.print_msg("\n", level=1)
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
