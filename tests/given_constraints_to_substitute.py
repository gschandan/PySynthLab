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

    def test_substitute_constraints_commutativity(self):
        self.problem.initialise_z3_variables()
        self.problem.initialise_z3_synth_functions()
        func = self.problem.context.z3_synth_functions["f"]
        args = [self.problem.context.z3_variables["x"], self.problem.context.z3_variables["y"]]

        candidate_expr, _ = self.problem.generate_max_function(args)
        substituted_constraints = self.problem.substitute_constraints(self.problem.context.z3_constraints, func,
                                                                      candidate_expr)

        self.assertGreater(len(substituted_constraints), 0)
        for constraint in substituted_constraints:
            self.assertIsInstance(constraint, BoolRef)
            self.assertTrue("f" not in str(constraint))

        commutativity_constraint = substituted_constraints[0]
        self.assertIn("x", str(commutativity_constraint))
        self.assertIn("y", str(commutativity_constraint))
        self.assertIn("==", str(commutativity_constraint))

    def test_substitute_constraints_additional(self):
        self.problem.initialise_z3_variables()
        self.problem.initialise_z3_synth_functions()
        func = self.problem.context.z3_synth_functions["f"]
        args = [self.problem.context.z3_variables["x"], self.problem.context.z3_variables["y"]]

        candidate_expr, _ = self.problem.generate_correct_abs_max_function(args)
        substituted_constraints = self.problem.substitute_constraints(self.problem.context.z3_constraints, func,
                                                                      candidate_expr)

        self.assertGreater(len(substituted_constraints), 0)
        for constraint in substituted_constraints:
            self.assertIsInstance(constraint, BoolRef)
            self.assertTrue("f" not in str(constraint))

        for constraint in substituted_constraints:
            self.assertTrue("If" in str(constraint))
            self.assertTrue("And" in str(constraint) or "Or" in str(constraint))

    def test_substitute_constraints_with_generated_expression(self):
        self.problem.initialise_z3_variables()
        self.problem.initialise_z3_synth_functions()
        func = self.problem.context.z3_synth_functions["f"]
        args = [self.problem.context.z3_variables["x"], self.problem.context.z3_variables["y"]]

        candidate_expr, _ = self.problem.generate_arithmetic_function(args, depth=2, complexity=3)
        substituted_constraints = self.problem.substitute_constraints(self.problem.context.z3_constraints, func,
                                                                      candidate_expr)

        self.assertGreater(len(substituted_constraints), 0)
        for constraint in substituted_constraints:
            self.assertIsInstance(constraint, BoolRef)
            self.assertTrue("f" not in str(constraint))

        for constraint in substituted_constraints:
            self.assertTrue(any(op in str(constraint) for op in ["+", "-", "*", "If"]))


if __name__ == "__main__":
    unittest.main()
