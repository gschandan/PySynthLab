import unittest
from z3 import *
from src.cegis.z3.synthesis_problem import SynthesisProblem
from src.utilities.options import Options
from src.cegis.z3.synthesis_strategy.synthesis_strategy import SynthesisStrategy
from src.cegis.z3.synthesis_strategy.random_search_top_down import RandomSearchStrategyTopDown
from src.cegis.z3.synthesis_strategy.random_search_bottom_up import RandomSearchStrategyBottomUp
from src.cegis.z3.synthesis_strategy.fast_enumerative_synth import FastEnumerativeSynthesis


class TestSynthesisStrategy(SynthesisStrategy):
    def execute_cegis(self) -> None:
        pass


class SimpleLIAProblem(unittest.TestCase):
    def setUp(self):
        self.problem_str = """
        (set-logic LIA)
        (define-fun double ((x Int)) Int (* x 2))
        (define-fun inc ((x Int)) Int (+ x 1))
        (synth-fun f ((x Int)) Int
            ((Start Int (x
                         0
                         1
                         (double Start)
                         (inc Start)
                         (+ Start Start)
                         (- Start Start)))))
        (declare-var x Int)
        (constraint (= (f x) (double (inc x))))
        (check-synth)
        """
        self.options = Options()
        self.options.synthesis_parameters.max_iterations = 5
        self.problem = SynthesisProblem(self.problem_str.strip(), self.options)

    def test_invalid_and_valid_solutions(self):
        x = Int('x')

        def incorrect_f(x):
            return x + 1

        incorrect_candidates = [incorrect_f(x)]
        incorrect_func_strs = ["incorrect_f"]

        strategy = TestSynthesisStrategy(self.problem)
        result = strategy.test_candidates(incorrect_func_strs, incorrect_candidates)
        self.assertFalse(result)

        def correct_f(x):
            return 2 * (x + 1)

        correct_candidates = [correct_f(x)]
        correct_func_strs = ["correct_f"]

        result = strategy.test_candidates(correct_func_strs, correct_candidates)
        self.assertTrue(result)

    def test_random_search_top_down_single_iteration(self):
        strategy = RandomSearchStrategyTopDown(self.problem)
        strategy.candidate_generator.max_depth = 2
        with self.assertLogs(self.problem.logger, level='INFO') as log_context:
            strategy.execute_cegis()

        log_messages = [log.message for log in log_context.records]
        self.assertTrue(any("Iteration 1/" in log for log in log_messages))
        self.assertTrue(any("Generated expression:" in log for log in log_messages))

    def test_random_search_bottom_up_single_iteration(self):
        strategy = RandomSearchStrategyBottomUp(self.problem)
        strategy.candidate_generator.max_depth = 2
        with self.assertLogs(self.problem.logger, level='INFO') as log_context:
            strategy.execute_cegis()

        log_messages = [log.message for log in log_context.records]
        self.assertTrue(any('Iteration 1/' in log for log in log_messages))
        self.assertTrue(any("Generated expression:" in log for log in log_messages))

    def test_fast_enumerative_synthesis_single_iteration(self):
        strategy = FastEnumerativeSynthesis(self.problem)
        strategy.candidate_generator.max_depth = 2
        with self.assertLogs(self.problem.logger, level='INFO') as log_context:
            strategy.execute_cegis()

        log_messages = [log.message for log in log_context.records]
        self.assertTrue(any('Depth' in log for log in log_messages))


if __name__ == '__main__':
    unittest.main()