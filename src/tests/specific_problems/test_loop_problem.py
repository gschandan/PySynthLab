import unittest
from typing import List, Tuple, Callable
from z3 import *
from src.cegis.z3.random_search_bottom_up import  SynthesisProblem
from src.cegis.z3.synthesis_problem import SynthesisProblemOptions


class LoopProblem(unittest.TestCase):
    def setUp(self):
        self.problem_str = """
        (set-logic LIA)
        (define-fun qm ((a Int) (b Int)) Int (ite (< a 0) b a))
        (synth-fun qm-loop ((x Int)) Int
        ((Start Int (x 0 1 3
        (- Start Start)
        (+ Start Start)
        (qm Start Start)))))
        (declare-var x Int)
        (constraint (= (qm-loop x) (ite (<= x 0) 3 (- x 1))))
        (check-synth)
        """
        self.options = SynthesisProblemOptions(verbose=0)
        self.problem = SynthesisProblem(self.problem_str, self.options)

    def test_valid_solution_is_correctly_identified(self):
        def generate_qm_loop_function(arg_sorts: List[z3.SortRef]) -> Tuple[Callable, str]:
            args = [z3.Var(i, sort) for i, sort in enumerate(arg_sorts)]

            def qm_loop_function(x):
                return z3.If(x <= 0, 3, x - 1)

            expr = qm_loop_function(*args)
            func_str = f"def qm_loop_function({', '.join(str(arg) for arg in args)}):\n"
            func_str += f"    return {str(expr)}\n"
            return qm_loop_function, func_str

        self.problem.print_msg("Trying known candidate for qm-loop")
        args = [list(self.problem.context.z3_synth_functions.values())[0].domain(i) for i in
                range(list(self.problem.context.z3_synth_functions.values())[0].arity())]
        candidate, func_str = generate_qm_loop_function(args)

        free_variables = [z3.Var(i, sort) for i, sort in enumerate(args)]
        candidate_function = candidate(*free_variables)

        self.problem.print_msg(f"candidate_function for substitution {candidate_function}")
        self.problem.print_msg(f"Testing guess: {func_str}")
        result = self.problem.test_candidates_alternative([func_str],
                                              [candidate_function])
        self.problem.print_msg("\n")
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()