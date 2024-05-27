import unittest
from z3 import *
from pysynthlab.cegis.z3.synthesis_problem_z3 import SynthesisProblemOptions, SynthesisProblem


class WhenTheConstraintIsCommutativity(unittest.TestCase):
    def setUp(self):
        self.problem_str = """
            (set-logic LIA)
            (synth-fun f ((x Int) (y Int)) Int)
            (declare-var x Int)
            (declare-var y Int)
            (constraint (= (f x y) (f y x)))
            (constraint (and (<= x (f x y)) (<= y (f x y))))
            (check-synth)
            """
        self.options = SynthesisProblemOptions()
        self.problem = SynthesisProblem(self.problem_str, self.options)
        self.problem.initialise_z3_variables()
        self.problem.initialise_z3_synth_functions()
        self.func = self.problem.context.z3_synth_functions["f"]
        self.args = [self.problem.context.z3_variables["x"], self.problem.context.z3_variables["y"]]

    def test_substitute_constraints_commutativity_with_invalid_function(self):
        candidate_expr, _ = self.problem.generate_non_commutative_solution(self.args)
        substituted_constraints = self.problem.substitute_constraints(self.problem.context.z3_constraints, self.func, candidate_expr)

        self.assertGreater(len(substituted_constraints), 0)

        for constraint in substituted_constraints:
            self.assertIsInstance(constraint, BoolRef)
            self.assertIn("If", str(constraint))
            self.assertIn("<=", str(constraint))

        # Check that the commutativity constraint has been correctly enforced
        commutativity_constraints = [str(c) for c in substituted_constraints if "==" in str(c)]
        self.assertGreater(len(commutativity_constraints), 0)

        for commutativity_constraint in commutativity_constraints:
            self.assertIn("x", commutativity_constraint)
            self.assertIn("y", commutativity_constraint)
            self.assertIn("==", commutativity_constraint)
            self.assertNotEqual(commutativity_constraint.count("If"), 1)

        # Check that the non-commutative function does not satisfy commutativity
        for constraint in substituted_constraints:
            self.assertNotEqual(str(constraint), "If(x > y, x, y - 1) == If(y > x, y, x - 1)")

    def test_substitute_constraints_commutativity(self):

        candidate_expr, _ = self.problem.generate_max_function(self.args)
        substituted_constraints = self.problem.substitute_constraints(
            self.problem.context.z3_constraints,
            self.func,
            candidate_expr
        )

        self.assertGreater(len(substituted_constraints), 0)

        commutativity_constraint = substituted_constraints[0]
        self.assertIsInstance(commutativity_constraint, BoolRef)
        self.assertIn("==", str(commutativity_constraint))

        other_constraints = substituted_constraints[1:]
        for constraint in other_constraints:
            self.assertIsInstance(constraint, BoolRef)
            self.assertIn("<=", str(constraint))
            self.assertIn("If", str(constraint))
            self.assertIn("x", str(constraint))
            self.assertIn("y", str(constraint))

        solver = Solver()
        solver.add(substituted_constraints)
        self.assertEqual(solver.check(), sat)


if __name__ == "__main__":
    unittest.main()
