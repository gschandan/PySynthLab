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


class LoopProblem(unittest.TestCase):
    def setUp(self):
        self.problem_str = """
        (set-logic LIA)
        (define-fun qm ((a Int) (b Int)) Int (ite (< a 0) b a))
        (synth-fun qm-loop ((x Int)) Int
        ((Start Int (x 0 1 3
        (- Start Start)
        (+ Start Start)
        (qm Start Start)))))
        (declare-var x Int)
        (constraint (= (qm-loop x) (ite (<= x 0) 3 (- x 1))))
        (check-synth)
        """
        self.options = Options()
        self.options.synthesis_parameters.max_iterations = 5
        self.problem = SynthesisProblem(self.problem_str.strip(), self.options)

    def test_invalid_and_valid_solutions(self):
        x = Int('x')

        def incorrect_qm_loop(x):
            return x + 1

        incorrect_candidates = [incorrect_qm_loop(x)]
        incorrect_func_strs = ["incorrect_qm_loop"]

        strategy = TestSynthesisStrategy(self.problem)
        result = strategy.test_candidates(incorrect_func_strs, incorrect_candidates)
        self.assertFalse(result)

        def correct_qm_loop(x):
            return If(x <= 0, 3, x - 1)

        correct_candidates = [correct_qm_loop(x)]
        correct_func_strs = ["correct_qm_loop"]

        result = strategy.test_candidates(correct_func_strs, correct_candidates)
        self.assertTrue(result)

    def test_random_search_top_down_single_iteration(self):
        strategy = RandomSearchStrategyTopDown(self.problem)
        strategy.candidate_generator.max_depth = 1
        with self.assertLogs(self.problem.logger, level='INFO') as log_context:
            strategy.execute_cegis()

        log_messages = [log.message for log in log_context.records]
        self.assertTrue(any("Iteration 1/" in log for log in log_messages))
        self.assertTrue(any("Generated expression:" in log for log in log_messages))

    def test_random_search_bottom_up_single_iteration(self):
        strategy = RandomSearchStrategyBottomUp(self.problem)
        strategy.candidate_generator.max_depth = 1
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
        self.assertTrue(any('Depth 0/' in log for log in log_messages))


if __name__ == '__main__':
    unittest.main()