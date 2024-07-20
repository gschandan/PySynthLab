import logging
import unittest
from contextlib import redirect_stdout

from z3 import *
from src.cegis.z3.synthesis_problem import SynthesisProblem
from src.utilities.options import Options
from src.cegis.z3.synthesis_strategy.synthesis_strategy import SynthesisStrategy
from src.cegis.z3.synthesis_strategy.random_search_top_down import RandomSearchStrategyTopDown
from src.cegis.z3.synthesis_strategy.random_search_bottom_up import RandomSearchStrategyBottomUp
from src.cegis.z3.synthesis_strategy.fast_enumerative_synth import FastEnumerativeSynthesis


class FourSynthFunctions(unittest.TestCase):
    def setUp(self):
        self.problem_str = """
            (set-logic LIA)
            (synth-fun id1 ((x Int)) Int (
                (Start Int (x (- Start) (+ Start x)))
            ))
            (synth-fun id2 ((x Int)) Int (
                (Start Int (x (- Start) (+ Start x)))
            ))
            (synth-fun id3 ((x Int)) Int (
                (Start Int (0 (- Start) (+ Start x)))
            ))
            (synth-fun id4 ((x Int)) Int (
                (Start Int (x (- Start) (+ Start x)))
            ))
            (declare-var x Int)
            (constraint (= (id1 x) (id2 x) (id3 x) (id4 x) x))
            (check-synth)
        """
        self.options = Options()
        self.options.synthesis_parameters.max_iterations = 5
        self.problem = SynthesisProblem(self.problem_str.strip(), self.options)

    def test_invalid_and_valid_solutions(self):
        x = Int('x')

        def incorrect_id1(x):
            return x + 1

        def incorrect_id2(x):
            return x - 1

        def incorrect_id3(x):
            return 0

        def incorrect_id4(x):
            return -x

        incorrect_candidates = [incorrect_id1(x), incorrect_id2(x), incorrect_id3(x), incorrect_id4(x)]
        incorrect_func_strs = ["incorrect_id1", "incorrect_id2", "incorrect_id3", "incorrect_id4"]

        strategy = SynthesisStrategy(self.problem)
        result = strategy.test_candidates(incorrect_func_strs, incorrect_candidates)
        self.assertFalse(result)

        def correct_id1(x):
            return x

        partially_correct_candidates = [correct_id1(x), incorrect_id2(x), incorrect_id3(x), incorrect_id4(x)]
        partially_correct_func_strs = ["correct_id1", "incorrect_id2", "incorrect_id3", "incorrect_id4"]

        result = strategy.test_candidates(partially_correct_func_strs, partially_correct_candidates)
        self.assertFalse(result)

        def correct_id2(x):
            return x

        def correct_id3(x):
            return x

        def correct_id4(x):
            return x

        correct_candidates = [correct_id1(x), correct_id2(x), correct_id3(x), correct_id4(x)]
        correct_func_strs = ["correct_id1", "correct_id2", "correct_id3", "correct_id4"]

        result = strategy.test_candidates(correct_func_strs, correct_candidates)
        self.assertTrue(result)

    def test_random_search_top_down(self):
        strategy = RandomSearchStrategyTopDown(self.problem)
        with self.assertLogs(self.problem.logger, level='INFO') as log_context:
            try:
                strategy.execute_cegis()
            except Exception as e:
                self.fail(f"execute_cegis() raised {type(e).__name__} unexpectedly: {str(e)}")

        self.assertTrue(any(record.levelno == logging.INFO for record in log_context.records))
        log_messages = [log.message for log in log_context.records]
        self.assertTrue(any("Iteration 1/5" in log for log in log_messages))
        self.assertTrue(any("Generated expression:" in log for log in log_messages))
        self.assertTrue(not all("Iteration 6/5" in log for log in log_messages))

    def test_random_search_bottom_up(self):
        strategy = RandomSearchStrategyBottomUp(self.problem)
        with self.assertLogs(self.problem.logger, level='INFO') as log_context:
            try:
                strategy.execute_cegis()
            except Exception as e:
                self.fail(f"execute_cegis() raised {type(e).__name__} unexpectedly: {str(e)}")

        self.assertTrue(any(record.levelno == logging.INFO for record in log_context.records))
        log_messages = [log.message for log in log_context.records]
        self.assertTrue(any('Iteration 1/5 depth: 1, complexity: 1, candidate at depth: 1/10)' in log for log in log_messages))
        self.assertTrue(any("Generated expression:" in log for log in log_messages))
        self.assertTrue(not all("Iteration 6/5" in log for log in log_messages))

    def test_fast_enumerative_synthesis(self):
        strategy = FastEnumerativeSynthesis(self.problem)
        with self.assertLogs(self.problem.logger, level='INFO') as log_context:
            try:
                strategy.execute_cegis()
            except Exception as e:
                self.fail(f"execute_cegis() raised {type(e).__name__} unexpectedly: {str(e)}")

        self.assertTrue(any(record.levelno == logging.INFO for record in log_context.records))
        log_messages = [log.message for log in log_context.records]
        self.assertTrue(any('Iteration 1/5' in log for log in log_messages))
        self.assertTrue(any("Generated expression:" in log for log in log_messages))
        self.assertTrue(not all("Iteration 6/5" in log for log in log_messages))


if __name__ == '__main__':
    unittest.main()
