import unittest
from typing import List, Tuple, Callable
from z3 import *
from src.cegis.z3.synthesis_strategy.random_search_bottom_up import SynthesisProblemZ3, RandomSearchStrategyBottomUp
from src.cegis.z3.synthesis_problem_z3 import Options
from src.helpers.parser.src.resolution import FunctionKind


class WhenTheProblemIsTheMaxOfTwoIntegersWithDifferentGlobalVariables(unittest.TestCase):
    def setUp(self):
        self.problem_str = """
        (set-logic LIA)
        (synth-fun max2 ((a Int) (b Int)) Int)
        (declare-var x Int)
        (declare-var y Int)
        (constraint (>= (max2 x y) x))
        (constraint (>= (max2 x y) y))
        (constraint (or (= x (max2 x y)) (= y (max2 x y))))
        (check-synth)
        """
        self.options = Options()
        self.problem = SynthesisProblemZ3(self.problem_str, self.options)

    def generate_max_function(self, arg_sorts: List[z3.SortRef]) -> Tuple[Callable, str]:
        args = [z3.Var(i, sort) for i, sort in enumerate(arg_sorts)]

        def max_function(*values):
            if len(values) != 2:
                raise ValueError("max_function expects exactly 2 arguments.")
            x, y = values
            return If(x <= y, y, x)

        expr = max_function(*args[:2])
        func_str = f"def max_function({', '.join(str(arg) for arg in args[:2])}):\n"
        func_str += f"    return {str(expr)}\n"
        return max_function, func_str

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
        self.problem.initialise_variables()
        self.assertIn("x", self.problem.context.z3_variables)
        self.assertIn("y", self.problem.context.z3_variables)
        self.assertIsInstance(self.problem.context.z3_variables["x"], ArithRef)
        self.assertIsInstance(self.problem.context.z3_variables["y"], ArithRef)

    def test_initialise_synth_functions(self):
        self.problem.initialise_synth_functions()
        self.assertIn("max2", self.problem.context.z3_synth_functions)
        self.assertIsInstance(self.problem.context.z3_synth_functions["max2"], FuncDeclRef)

    def test_parse_constraints(self):
        self.problem.initialise_variables()
        self.problem.initialise_synth_functions()
        self.problem.parse_constraints()
        self.assertGreater(len(self.problem.context.z3_constraints), 0)
        self.assertIsInstance(self.problem.context.z3_constraints[0], BoolRef)

    def test_substitute_constraints_multiple(self):
        self.problem.initialise_variables()
        self.problem.initialise_synth_functions()
        self.problem.parse_constraints()
        constraints = self.problem.context.z3_constraints
        func = self.problem.context.z3_synth_functions["max2"]
        args = [self.problem.context.z3_variables["x"], self.problem.context.z3_variables["y"]]
        candidate_expr, _ = self.generate_max_function([IntSort(), IntSort()])
        candidate_func = candidate_expr(*args)
        substituted_constraints = self.problem.substitute_constraints(constraints, [func], [candidate_func])
        self.assertGreater(len(substituted_constraints), 0)
        self.assertIsInstance(substituted_constraints[0], BoolRef)

    def test_test_multiple_candidates(self):
        self.problem.initialise_variables()
        self.problem.initialise_synth_functions()
        self.problem.map_z3_variables()
        self.problem.parse_constraints()
        args = [self.problem.context.z3_variables["x"], self.problem.context.z3_variables["y"]]
        candidate_expr, func_str = self.generate_max_function([IntSort(), IntSort()])
        candidate_function = candidate_expr(*args)
        strategy = RandomSearchStrategyBottomUp(self.problem)
        result = strategy.test_candidates([func_str], [candidate_function])
        self.assertTrue(result)

    def test_get_logic(self):
        logic = self.problem.get_logic()
        self.assertEqual(logic, "LIA")

    def test_get_synth_funcs(self):
        synth_funcs = self.problem.get_synth_funcs()
        self.assertEqual(list(synth_funcs.values())[0].identifier.symbol, "max2")
        self.assertListEqual(list(synth_funcs.values())[0].argument_names, ["a", "b"])
        self.assertEqual(list(synth_funcs.values())[0].function_kind, FunctionKind.SYNTH_FUN)

    def test_get_var_symbols(self):
        var_symbols = self.problem.get_var_symbols()
        self.assertIn("x", var_symbols)
        self.assertIn("y", var_symbols)
