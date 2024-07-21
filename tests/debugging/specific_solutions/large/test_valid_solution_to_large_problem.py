import unittest
from typing import List, Tuple, Callable
from z3 import *
from src.cegis.z3.synthesis_strategy.random_search_bottom_up import  SynthesisProblem
from src.utilities.options import Options
from tests.helpers.SynthesisStrategyHelper import TestSynthesisStrategy


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
        self.options = Options()
        self.problem = SynthesisProblem(self.problem_str, self.options)
        self.strategy = TestSynthesisStrategy(self.problem)

    def test_valid_solution_is_correctly_identified(self):
        def generate_valid_solution(arg_sorts: List[z3.SortRef]) -> Tuple[Callable, str]:
            free_variables = [z3.Var(i, sort) for i, sort in enumerate(arg_sorts)]
            def valid_function(*values):
                x, y = values
                return If(x <= y, (100 * x) + 1000, (100 * y) + 1000)

            expr = valid_function(*free_variables)
            func_str = f"def valid_function({', '.join(str(arg) for arg in free_variables)}):\n"
            func_str += f"    return {str(expr)}\n"
            return valid_function, func_str

        args = [list(self.problem.context.z3_synth_functions.values())[0].domain(i) for i in
                range(list(self.problem.context.z3_synth_functions.values())[0].arity())]
        func = self.problem.context.z3_synth_functions["f"]
        candidate, func_str = generate_valid_solution(args)

        free_variables = [z3.Var(i, sort) for i, sort in enumerate(args)]
        candidate_function = candidate(*free_variables)

        with self.assertLogs(self.problem.logger, level='INFO') as log_context:
            result = self.strategy.test_candidates([func_str],
                                                   [candidate_function])

        log_messages = [log.message for log in log_context.records]
        self.assertTrue(any("No counterexample found for candidates [(If(Var(0) <= Var(1), 100*Var(0) + 1000, 100*Var(1) + 1000), 'f')]" in log for log in log_messages))
        self.assertTrue(any("No counterexample found! Guess(es) should be correct: def valid_function(Var(0), Var(1)):" in log for log in log_messages))
        self.assertTrue(any("return If(Var(0) <= Var(1), 100*Var(0) + 1000, 100*Var(1) + 1000)" in log for log in log_messages))

        self.assertTrue(result)

        substituted_constraints = self.problem.substitute_constraints(self.problem.context.z3_constraints,
                                                                      [func],
                                                                      [candidate_function])
        self.assertGreater(len(substituted_constraints), 0)

    def test_invalid_solution_is_correctly_identified(self):

        def generate_invalid_solution(arg_sorts: List[z3.SortRef]) -> Tuple[Callable, str]:
            free_variables = [z3.Var(i, sort) for i, sort in enumerate(arg_sorts)]

            def valid_function(*values):
                x, y = values
                return If(x <= y, 100 + 1000, (100 * y) + 1000)

            expr = valid_function(*free_variables)
            func_str = f"def valid_function({', '.join(str(arg) for arg in free_variables)}):\n"
            func_str += f"    return {str(expr)}\n"
            return valid_function, func_str

        args = [list(self.problem.context.z3_synth_functions.values())[0].domain(i) for i in
                range(list(self.problem.context.z3_synth_functions.values())[0].arity())]
        func = self.problem.context.z3_synth_functions["f"]
        candidate, func_str = generate_invalid_solution(args)

        free_variables = [z3.Var(i, sort) for i, sort in enumerate(args)]
        candidate_function = candidate(*free_variables)

        with self.assertLogs(self.problem.logger, level='INFO') as log_context:
            result = self.strategy.test_candidates([func_str],
                                                   [candidate_function])

        log_messages = [log.message for log in log_context.records]
        self.assertTrue(any("Counterexample for f:" in log for log in log_messages))
        self.assertTrue(any("New counterexample found for f" in log for log in log_messages))
        self.assertTrue(any("Candidates: [(If(Var(0) <= Var(1), 1100, 100*Var(1) + 1000), 'f')]" in log for log in log_messages))


        self.assertFalse(result)

        substituted_constraints = self.problem.substitute_constraints(self.problem.context.z3_constraints,
                                                                      [func],
                                                                      [candidate_function])
        self.assertGreater(len(substituted_constraints), 0)



if __name__ == '__main__':
    unittest.main()