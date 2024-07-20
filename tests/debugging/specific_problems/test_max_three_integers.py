import unittest
from typing import List, Tuple, Callable
from z3 import *
from src.cegis.z3.synthesis_strategy.random_search_bottom_up import  SynthesisProblem
from src.cegis.z3.synthesis_problem import Options

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
        self.options = Options()
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

        SynthesisProblem.logger.info("Trying known candidate for max 3")
        args = [list(self.problem.context.z3_synth_functions.values())[0].domain(i) for i in
                range(list(self.problem.context.z3_synth_functions.values())[0].arity())]
        candidate, func_str = generate_max_function(args)

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
