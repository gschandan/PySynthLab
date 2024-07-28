from abc import abstractmethod, ABC
from typing import List, Tuple
from z3 import *
from src.cegis.z3.synthesis_problem import SynthesisProblem


class CandidateGenerator(ABC):
    def __init__(self, problem: 'SynthesisProblem'):
        self.problem = problem
        self.config = problem.options
        self.min_const = SynthesisProblem.options.synthesis_parameters.min_const
        self.max_const = SynthesisProblem.options.synthesis_parameters.max_const
        self.operation_costs = SynthesisProblem.options.synthesis_parameters.operation_costs

    @abstractmethod
    def generate_candidates(self) -> List[Tuple[z3.ExprRef, str]]:
        """Generate candidate functions."""
        pass

    @abstractmethod
    def prune_candidates(self, candidates: List[Tuple[z3.ExprRef, str]]) -> List[Tuple[z3.ExprRef, str]]:
        """Prune candidate functions."""
        pass

    def op_complexity(self, op: str) -> int:
        return self.operation_costs.get(op, 0)

    def get_arg_sorts(self, func_name: str) -> List[z3.SortRef]:
        func = self.problem.context.z3_synth_functions[func_name]
        return [func.domain(i) for i in range(func.arity())]

    @staticmethod
    def create_candidate_function(candidate_expr: z3.ExprRef, arg_sorts: List[z3.SortRef]) -> z3.ExprRef:
        args = [z3.Var(i, sort) for i, sort in enumerate(arg_sorts)]
        return z3.substitute(candidate_expr, [(arg, z3.Var(i, arg.sort())) for i, arg in enumerate(args)])
