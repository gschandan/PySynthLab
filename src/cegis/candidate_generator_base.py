from abc import ABC, abstractmethod

from src.cegis.synthesis_problem_base import BaseSynthesisProblem


# TODO refactor/abstract to allow swapping out z3 for cvc5
class AbstractCandidateGenerator(ABC):
    @abstractmethod
    def __init__(self, problem: BaseSynthesisProblem):
        pass

    @abstractmethod
    def generate_candidates(self):
        pass

    @abstractmethod
    def generate_random_term(self, arg_sorts, depth, complexity, operations=None):
        pass

    @abstractmethod
    def generate_condition(self, args):
        pass

    @abstractmethod
    def prune_candidates(self, candidates):
        pass

