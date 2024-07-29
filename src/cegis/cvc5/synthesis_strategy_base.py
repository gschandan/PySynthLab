from abc import ABC, abstractmethod
from typing import Tuple
from cvc5 import Term
from src.cegis.cvc5.candidate_generator_base import CandidateGeneratorCVC5
from src.cegis.cvc5.random_candidate_generator import RandomCandidateGeneratorCVC5
from src.cegis.cvc5.synthesis_problem_cvc5 import SynthesisProblemCVC5


# TODO refactor/abstract to allow swapping out z3 for cvc5
class SynthesisStrategyCVC5(ABC):
    """
    An abstract base class for implementing various synthesis strategies using CVC5.
    """

    def __init__(self, problem: 'SynthesisProblemCVC5', candidate_generator: 'CandidateGeneratorCVC5' = None):
        self.problem = problem
        self.solution_found = False
        if candidate_generator is None:
            self.candidate_generator = RandomCandidateGeneratorCVC5(problem)
        else:
            self.candidate_generator = candidate_generator

    def generate_candidates(self) -> list[Tuple[Term, str]]:
        return self.candidate_generator.generate_candidates()

    @abstractmethod
    def execute_cegis(self) -> None:
        pass

    def set_solution_found(self) -> None:
        self.solution_found = True

    def set_candidate_generator(self, generator: 'CandidateGeneratorCVC5'):
        self.candidate_generator = generator

    def test_candidates(self, func_strs: list[str], candidate_functions: list[Term]) -> bool:
        pass

    def check_counterexample(self, func_name: str, candidate: Term) -> bool:
        pass

    def generate_counterexample(self, candidates: list[Tuple[Term, str]]) -> dict[str, dict[str, Term]] | None:
        pass

    def verify_candidates(self, candidates: list[Term]) -> bool:
        pass
