from abc import abstractmethod, ABC
from typing import  List, Tuple

from z3 import *

from src.cegis.z3.options import Options
from src.cegis.z3.synthesis_problem import SynthesisProblem


class CandidateGenerator(ABC):
    def __init__(self, problem: 'SynthesisProblem', config: Options):
        self.problem = problem
        self.config = config
        self.min_const = config.synthesis_parameters_min_const
        self.max_const = config.synthesis_parameters_max_const

    @abstractmethod
    def generate_candidates(self) -> List[Tuple[z3.ExprRef, str]]:
        """Generate candidate functions."""
        pass

    @abstractmethod
    def prune_candidates(self, candidates: List[Tuple[z3.ExprRef, str]]) -> List[Tuple[z3.ExprRef, str]]:
        """Prune candidate functions."""
        pass

    def get_arg_sorts(self, func_name: str) -> List[z3.SortRef]:
        func = self.problem.context.z3_synth_functions[func_name]
        return [func.domain(i) for i in range(func.arity())]
