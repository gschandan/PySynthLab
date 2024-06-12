import unittest
from typing import List, Tuple, Callable
from z3 import *
from src.cegis.z3.synthesis_problem_z3 import SynthesisProblemOptions, SynthesisProblem


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
        self.options = SynthesisProblemOptions()
        self.problem = SynthesisProblem(self.problem_str, self.options)

    def test_valid_solution_is_correctly_identified(self):
        def generate_valid_solution(arg_sorts: List[z3.SortRef]) -> Tuple[Callable, str]:
            args = [z3.Var(i, sort) for i, sort in enumerate(arg_sorts)]
            def valid_function(*values):
                x, y = values
                return If(x <= y, 10 * y + 100, 10 * x + 100)

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
        result = self.problem.test_multiple_candidates(self.problem.context.z3_constraints,
                                                       self.problem.context.negated_constraints, [func_str],
                                                       [candidate_function], [args])
        print(self.problem.context.verification_solver.to_smt2())
        print(self.problem.context.enumerator_solver.to_smt2())
        
        self.assertTrue(result)

        substituted_constraints = self.problem.substitute_constraints_multiple(self.problem.context.z3_constraints,
                                                                               [func], 
                                                                               [candidate_function])
        self.assertGreater(len(substituted_constraints), 0)

        expected_commutativity = And(
            If(self.args[0] <= self.args[1], 10 * self.args[1] + 100, 10 * self.args[0] + 100) ==
            If(self.args[1] <= self.args[0], 10 * self.args[0] + 100, 10 * self.args[1] + 100)
        )

        expected_func_constraints = And(
            self.problem.context.z3_predefined_functions["func"](self.args[0]) >=
            If(self.args[0] <= self.args[1], 10 * self.args[1] + 100, 10 * self.args[0] + 100),
            self.problem.context.z3_predefined_functions["func"](self.args[1]) >=
            If(self.args[0] <= self.args[1], 10 * self.args[1] + 100, 10 * self.args[0] + 100)
        )

        commutativity_constraints = [c for c in substituted_constraints if "==" in str(c)]
        self.assertGreater(len(commutativity_constraints), 0)
        self.assertIn(str(expected_commutativity), [str(c) for c in commutativity_constraints])

        func_constraints = [c for c in substituted_constraints if ">=" in str(c)]
        self.assertGreater(len(func_constraints), 0)
        for constraint in func_constraints:
            self.assertIn(str(constraint), str(expected_func_constraints))



if __name__ == '__main__':
    unittest.main()