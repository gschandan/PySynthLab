import unittest
from z3 import *

from src.cegis.z3.synthesis_problem_z3 import SynthesisProblemOptions, SynthesisProblem
from src.helpers.parser.src.resolution import FunctionKind


class WhenTheProblemISTheMaxOfTwoIntegers(unittest.TestCase):
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

    def test_initialization(self):
        self.assertEqual(self.problem.input_problem, self.problem_str)
        self.assertIsNotNone(self.problem.symbol_table)
        self.assertGreater(len(self.problem.symbol_table.synth_functions), 0)
        self.assertGreater(len(self.problem.context.constraints), 0)

    def test_convert_sygus_to_smt(self):
        smt_problem = self.problem.convert_sygus_to_smt()
        self.assertIn("(assert", smt_problem)
        self.assertIn("(declare-fun", smt_problem)

    def test_initialise_z3_variables(self):
        self.problem.initialise_z3_variables()
        self.assertIn("x", self.problem.context.z3_variables)
        self.assertIn("y", self.problem.context.z3_variables)
        self.assertIsInstance(self.problem.context.z3_variables["x"], ArithRef)
        self.assertIsInstance(self.problem.context.z3_variables["y"], ArithRef)

    def test_initialise_z3_synth_functions(self):
        self.problem.initialise_z3_synth_functions()
        self.assertIn("f", self.problem.context.z3_synth_functions)
        self.assertIsInstance(self.problem.context.z3_synth_functions["f"], FuncDeclRef)

    def test_parse_constraints(self):
        self.problem.parse_constraints()
        self.assertGreater(len(self.problem.context.z3_constraints), 0)
        self.assertIsInstance(self.problem.context.z3_constraints[0], BoolRef)

    def test_substitute_constraints(self):
        self.problem.initialise_z3_variables()
        self.problem.initialise_z3_synth_functions()
        constraints = self.problem.context.z3_constraints
        func = self.problem.context.z3_synth_functions["f"]
        candidate_expr = self.problem.generate_max_function(
            [self.problem.context.z3_variables["x"], self.problem.context.z3_variables["y"]])[0]
        substituted_constraints = self.problem.substitute_constraints(constraints, func, candidate_expr)
        self.assertGreater(len(substituted_constraints), 0)
        self.assertIsInstance(substituted_constraints[0], BoolRef)

    def test_test_candidate(self):
        self.problem.initialise_z3_variables()
        self.problem.initialise_z3_synth_functions()
        self.problem.parse_constraints()
        func = self.problem.context.z3_synth_functions["f"]
        args = [self.problem.context.z3_variables["x"], self.problem.context.z3_variables["y"]]
        candidate_expr, func_str = self.problem.generate_max_function(args)
        result = self.problem.test_candidate(self.problem.context.z3_constraints, self.problem.context.negated_assertions, func_str, func, args, candidate_expr)
        self.assertTrue(result)

    def test_get_logic(self):
        logic = self.problem.get_logic()
        self.assertEqual(logic, "LIA")

    def test_get_synth_funcs(self):
        synth_funcs = self.problem.get_synth_funcs()
        self.assertEqual("f", list(synth_funcs.items())[0][0].symbol)
        self.assertListEqual(['x','y'], list(synth_funcs.items())[0][1].argument_names)
        self.assertEqual(list(synth_funcs.items())[0][1].function_kind.name, FunctionKind.SYNTH_FUN.name)

    def test_get_var_symbols(self):
        var_symbols = self.problem.get_var_symbols()
        self.assertIn("x", var_symbols)
        self.assertIn("y", var_symbols)


if __name__ == "__main__":
    unittest.main()
