from abc import ABC, abstractmethod

from src.cegis.synthesis_problem_base import BaseSynthesisProblem


# TODO refactor/abstract to allow swapping out z3 for cvc5
class AbstractSynthesisStrategy(ABC):
    @abstractmethod
    def __init__(self, problem: BaseSynthesisProblem):
        pass

    @abstractmethod
    def execute_cegis(self):
        pass

    @abstractmethod
    def test_candidates(self, func_strs, candidate_functions):
        pass
