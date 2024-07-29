from abc import ABC, abstractmethod
from typing import Tuple

from cvc5 import Sort, Term


class CandidateGeneratorCVC5(ABC):
    """
    Abstract base class for candidate function generators using CVC5.
    """

    def __init__(self, problem: 'SynthesisProblemCVC5'):
        self.problem = problem
        self.config = problem.options
        self.min_const = self.config.synthesis_parameters.min_const
        self.max_const = self.config.synthesis_parameters.max_const
        self.operation_costs = self.config.synthesis_parameters.operation_costs

    @abstractmethod
    def generate_candidates(self) -> list[Tuple[Term, str]]:
        pass

    @abstractmethod
    def prune_candidates(self, candidates: list[Tuple[Term, str]]) -> list[Tuple[Term, str]]:
        pass

    @staticmethod
    def create_candidate_function(candidate_expr: Term, arg_sorts: list[Sort]) -> Term:
        #TODO
        pass

    def op_complexity(self, op: str) -> int:
        return self.operation_costs.get(op, 0)

    def get_arg_sorts(self, func_name: str) -> list[Sort]:
        func = self.problem.cvc5_synth_functions[func_name]
        return func.getSort().getFunctionDomainSorts()

