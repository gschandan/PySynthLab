import unittest
from src.cegis.z3.synthesis_problem_z3 import SynthesisProblemZ3
from src.utilities.options import Options
from src.cegis.z3.synthesis_strategy.random_search_top_down import RandomSearchStrategyTopDown
from src.cegis.z3.synthesis_strategy.random_search_bottom_up import RandomSearchStrategyBottomUp
from abc import ABC, abstractmethod
from tests.helpers.SynthesisStrategyHelper import TestSynthesisStrategy


class SynthesisProblemTestCase(unittest.TestCase, ABC):
    problem_str = None

    @classmethod
    def setUpClass(cls):
        if cls.problem_str is None:
            raise NotImplementedError("Subclasses must define problem_str")
        cls.options = Options()
        cls.options.synthesis_parameters.max_iterations = 5
        cls.problem = SynthesisProblemZ3(cls.problem_str.strip(), cls.options)

    def setUp(self):
        self.test_strategy = TestSynthesisStrategy(self.problem)

    @abstractmethod
    def test_invalid_and_valid_solutions(self):
        pass

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

