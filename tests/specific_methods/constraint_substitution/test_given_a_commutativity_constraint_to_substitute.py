import typing
import unittest
from typing import List, Tuple, Callable, Collection
from z3 import *
from src.cegis.z3.synthesis_strategy.random_search_bottom_up import SynthesisProblem
from src.cegis.z3.synthesis_problem import Options


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
        self.options = Options()
        self.problem = SynthesisProblem(self.problem_str, self.options)

    def substitute_constraints(self, constraints: Collection[z3.ExprRef],
                               functions_to_replace: List[z3.FuncDeclRef],
                               candidate_functions: List[
                                   typing.Union[z3.FuncDeclRef, z3.QuantifierRef, z3.ExprRef, Callable]]) -> \
            List[z3.ExprRef]:
        """
        Substitute candidate expressions into a list of constraints.
        """
        synth_substitutions = list(zip(functions_to_replace, candidate_functions))
        predefined_substitutions = [(func, body) for func, body in self.problem.context.z3_predefined_functions.values()]

        substituted_constraints = []
        for constraint in constraints:
            synth_substituted = z3.substitute_funs(constraint, synth_substitutions)
            predefined_substituted = z3.substitute_funs(synth_substituted, predefined_substitutions)
            substituted_constraints.append(predefined_substituted)
        return substituted_constraints

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
