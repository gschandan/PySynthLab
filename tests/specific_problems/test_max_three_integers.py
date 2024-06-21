import unittest
from typing import List, Tuple, Callable
from z3 import *
from src.cegis.z3.bottom_up_random_search import  SynthesisProblem
from src.cegis.z3.synthesis_problem import SynthesisProblemOptions

class MaxThreeIntegers(unittest.TestCase):
    def setUp(self):
        self.problem_str = """
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
        self.options = SynthesisProblemOptions()
        self.problem = SynthesisProblem(self.problem_str, self.options)

    def test_valid_solutions_are_correctly_identified(self):
        def generate_max_function(arg_sorts: List[z3.SortRef]) -> Tuple[Callable, str]:
            args = [z3.Var(i, sort) for i, sort in enumerate(arg_sorts)]

            def max_function_3(*values):
                x, y, z = values
                return z3.If(x >= y, z3.If(x >= z, x, z), z3.If(y >= z, y, z))

            expr = max_function_3(*args)
            func_str = f"def max_3_function({', '.join(str(arg) for arg in args)}):\n"
            func_str += f"    return {str(expr)}\n"
            return max_function_3, func_str

        self.problem.print_msg("Trying known candidate for max 3", level=0)
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
