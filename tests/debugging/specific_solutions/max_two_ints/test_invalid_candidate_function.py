import unittest
from typing import List, Tuple, Callable
from z3 import *
from src.cegis.z3.synthesis_strategy.random_search_bottom_up import  SynthesisProblem
from src.cegis.z3.synthesis_problem import Options
from tests.helpers.SynthesisStrategyHelper import TestSynthesisStrategy


class GivenTheMaxOfTwoIntegersProblem(unittest.TestCase):
    def setUp(self):
        self.problem_str = """
        (synth-fun f ((x Int) (y Int)) Int)
        (declare-var x Int)
        (declare-var y Int)
        (constraint (= (f x y) (f y x)))
        (constraint (and (<= x (f x y)) (<= y (f x y))))
        """
        self.options = Options()
        self.problem = SynthesisProblem(self.problem_str, self.options)
        self.strategy = TestSynthesisStrategy(self.problem)

    def test_valid_solutions_Are_correctly_identified(self):
        def generate_invalid_solution_two(arg_sorts: List[z3.SortRef]) -> Tuple[Callable, str]:
            args = [z3.Var(i, sort) for i, sort in enumerate(arg_sorts)]

            def invalid_function(*values):
                if len(values) != 2:
                    raise ValueError("invalid_function expects exactly 2 arguments.")
                x, y = values
                return If(x > y, x, y - 1)

            expr = invalid_function(*args[:2])
            func_str = f"def invalid_function_two({', '.join(str(arg) for arg in args[:2])}):\n"
            func_str += f"    return {str(expr)}\n"
            return invalid_function, func_str

        SynthesisProblem.logger.info("Trying known invalid candidate for max")
        func_name = list(self.problem.context.z3_synth_functions.keys())[0]
        func = self.problem.context.z3_synth_functions[func_name]
        args = [func.domain(i) for i in range(func.arity())]
        candidate, func_str = generate_invalid_solution_two(args)

        variable_mapping = self.problem.context.variable_mapping_dict[func_name]
        free_variables = list(variable_mapping.keys())
        candidate_function = candidate(*free_variables)

        with self.assertLogs(self.problem.logger, level='INFO') as log_context:
            result = self.strategy.test_candidates([func_str],
                                                   [candidate_function])

        log_messages = [log.message for log in log_context.records]
        self.assertTrue(any("Counterexample for f:" in log for log in log_messages))
        self.assertTrue(any("New counterexample found for f:" in log for log in log_messages))
        self.assertTrue(any("Candidates: [(If(Var(0) > Var(1), Var(0), Var(1) - 1), 'f'" in log for log in log_messages))
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()