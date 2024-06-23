import unittest
from typing import List, Tuple, Callable
from z3 import *
from src.cegis.z3.random_search_bottom_up import  SynthesisProblem
from src.cegis.z3.synthesis_problem import SynthesisProblemOptions


class GivenAValidSolutionToTheLargeProblem(unittest.TestCase):
    def setUp(self):
        self.problem_str = """
        (set-logic LIA)
        (synth-fun f ((x Int) (y Int)) Int)
        (define-fun func ((x Int)) Int (+ (* x 100) 1000))
        (declare-var x Int)
        (declare-var y Int)
        (constraint (= (f x y) (f y x)))
        (constraint (and (>= (func x) (f x y)) (>= (func y) (f x y))))
        (check-synth)
        """
        self.options = SynthesisProblemOptions(verbose=0)
        self.problem = SynthesisProblem(self.problem_str, self.options)

    def test_valid_solution_is_correctly_identified(self):
        def generate_valid_solution(arg_sorts: List[z3.SortRef]) -> Tuple[Callable, str]:
            args = [z3.Var(i, sort) for i, sort in enumerate(arg_sorts)]
            def valid_function(*values):
                x, y = values
                return If(x <= y, (100 * x) + 1000, (100 * y) + 1000)

            expr = valid_function(*args)
            func_str = f"def valid_function({', '.join(str(arg) for arg in args)}):\n"
            func_str += f"    return {str(expr)}\n"
            return valid_function, func_str

        print("Trying valid candidate")
        args = [list(self.problem.context.z3_synth_functions.values())[0].domain(i) for i in
                range(list(self.problem.context.z3_synth_functions.values())[0].arity())]
        func = self.problem.context.z3_synth_functions["f"]
        candidate, func_str = generate_valid_solution(args)

        free_variables = [z3.Var(i, sort) for i, sort in enumerate(args)]
        candidate_function = candidate(*free_variables)

        print(f"candidate_function for substitution {candidate_function}")
        print(f"Testing guess: {func_str}")
        result = self.problem.test_candidates([func_str],
                                              [candidate_function])
        print(self.problem.context.verification_solver.to_smt2())
        print(self.problem.context.enumerator_solver.to_smt2())
        
        self.assertTrue(result)

        substituted_constraints = self.problem.substitute_constraints(self.problem.context.z3_constraints,
                                                                      [func],
                                                                      [candidate_function])
        self.assertGreater(len(substituted_constraints), 0)



if __name__ == '__main__':
    unittest.main()