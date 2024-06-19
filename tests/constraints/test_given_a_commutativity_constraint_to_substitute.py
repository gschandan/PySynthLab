import unittest
from typing import List, Tuple, Callable
from z3 import *
from src.cegis.z3.synthesis_problem_z3 import SynthesisProblemOptions, SynthesisProblem


class WhenTheConstraintIsCommutativity(unittest.TestCase):
    def setUp(self):
        self.problem_str = """
            (set-logic LIA)
            (synth-fun f ((x Int) (y Int)) Int)
            (declare-var x Int)
            (declare-var y Int)
            (constraint (= (f x y) (f y x)))
            (check-synth)
            """
        self.options = SynthesisProblemOptions()
        self.problem = SynthesisProblem(self.problem_str, self.options)

    def test_substitute_constraints_multiple_commutativity(self):
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

        constraints = self.problem.context.z3_constraints
        func = self.problem.context.z3_synth_functions["f"]
        candidate_expr, _ = generate_correct_abs_max_function([IntSort(), IntSort()])
        args = [self.problem.context.z3_variables["x"], self.problem.context.z3_variables["y"]]
        candidate_func = candidate_expr(*args)
        substituted_constraints = self.problem.substitute_constraints(constraints, [func], [candidate_func])
        self.assertGreater(len(substituted_constraints), 0)

        expected_commutativity = And(
            If(If(args[0] >= 0, args[0], -args[0]) > If(args[1] >= 0, args[1], -args[1]),
               If(args[0] >= 0, args[0], -args[0]),
               If(args[1] >= 0, args[1], -args[1])) ==
            If(If(args[0] >= 0, args[0], -args[0]) > If(args[1] >= 0, args[1], -args[1]),
               If(args[0] >= 0, args[0], -args[0]),
               If(args[1] >= 0, args[1], -args[1]))
        )

        for constraint in substituted_constraints:
            self.assertIsInstance(constraint, BoolRef)
            self.assertIn("If", str(constraint))

        commutativity_constraints = [c for c in substituted_constraints if "==" in str(c)]
        self.assertGreater(len(commutativity_constraints), 0)
        self.assertIn(str(expected_commutativity), [str(c) for c in commutativity_constraints])


if __name__ == "__main__":
    unittest.main()
