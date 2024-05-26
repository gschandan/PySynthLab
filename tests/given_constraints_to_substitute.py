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

    def test_substitute_constraints_commutativity_with_max_function(self):
        candidate_expr, _ = self.problem.generate_max_function(self.args)
        substituted_constraints = self.problem.substitute_constraints(self.problem.context.z3_constraints, self.func,
                                                                      candidate_expr)

        self.assertGreater(len(substituted_constraints), 0)
        for constraint in substituted_constraints:
            self.assertIsInstance(constraint, BoolRef)
            self.assertIn("If", str(constraint))
            self.assertIn("<=", str(constraint))

        commutativity_constraint = substituted_constraints[0]
        self.assertIn("x", str(commutativity_constraint))
        self.assertIn("y", str(commutativity_constraint))
        self.assertIn("==", str(commutativity_constraint))

    def test_substitute_constraints_commutativity_with_abs_max_function(self):
        candidate_expr, _ = self.problem.generate_correct_abs_max_function(self.args)
        substituted_constraints = self.problem.substitute_constraints(self.problem.context.z3_constraints, self.func,
                                                                      candidate_expr)

        self.assertGreater(len(substituted_constraints), 0)
        for constraint in substituted_constraints:
            self.assertIsInstance(constraint, BoolRef)
            self.assertIn("If", str(constraint))
            self.assertIn("<=", str(constraint))

        commutativity_constraint = substituted_constraints[0]
        self.assertIn("x", str(commutativity_constraint))
        self.assertIn("y", str(commutativity_constraint))
        self.assertIn("==", str(commutativity_constraint))

    def test_substitute_constraints_commutativity_with_invalid_function(self):
        candidate_expr, _ = self.problem.generate_non_commutative_solution(self.args)
        substituted_constraints = self.problem.substitute_constraints(self.problem.context.z3_constraints, self.func,
                                                                      candidate_expr)

        self.assertGreater(len(substituted_constraints), 0)
        for constraint in substituted_constraints:
            self.assertIsInstance(constraint, BoolRef)
            self.assertIn("If", str(constraint))
            self.assertIn("<=", str(constraint))

        commutativity_constraint = substituted_constraints[0]
        self.assertIn("x", str(commutativity_constraint))
        self.assertIn("y", str(commutativity_constraint))
        self.assertNotIn("==", str(commutativity_constraint))

    def test_substitute_constraints_commutativity_with_integer(self):
        candidate_expr = 1
        substituted_constraints = self.problem.substitute_constraints(self.problem.context.z3_constraints, self.func,
                                                                      candidate_expr)

        self.assertGreater(len(substituted_constraints), 0)
        for constraint in substituted_constraints:
            self.assertIsInstance(constraint, BoolRef)
            self.assertIn("<=", str(constraint))

        commutativity_constraint = substituted_constraints[0]
        self.assertIn("x", str(commutativity_constraint))
        self.assertIn("y", str(commutativity_constraint))
        self.assertIn("==", str(commutativity_constraint))

    def test_substitute_constraints_commutativity_with_arithmetic_expression(self):
        candidate_expr, _ = self.problem.generate_arithmetic_function(self.args, depth=2, complexity=3)
        substituted_constraints = self.problem.substitute_constraints(self.problem.context.z3_constraints, self.func,
                                                                      candidate_expr)

        self.assertGreater(len(substituted_constraints), 0)
        for constraint in substituted_constraints:
            self.assertIsInstance(constraint, BoolRef)
            self.assertTrue(any(op in str(constraint) for op in ["+", "-", "*", "If"]))

        commutativity_constraint = substituted_constraints[0]
        self.assertIn("x", str(commutativity_constraint))
        self.assertIn("y", str(commutativity_constraint))
        self.assertIn("==", str(commutativity_constraint))


if __name__ == "__main__":
    unittest.main()