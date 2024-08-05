from abc import ABC, abstractmethod
from typing import List, Tuple
import z3

from src.cegis.z3.synthesis_problem_z3 import SynthesisProblemZ3
from src.utilities.cancellation_token import GlobalCancellationToken


class CandidateGenerator(ABC):
    """
    Abstract base class for candidate function generators.

    This class defines the interface for generating and pruning candidate functions
    in the context of program synthesis.

    Attributes:
        problem (SynthesisProblemZ3): The synthesis problem instance.
        config (object): Configuration options for the synthesis problem.
        min_const (int): Minimum constant value allowed in candidate functions.
        max_const (int): Maximum constant value allowed in candidate functions.
        operation_costs (dict): Dictionary mapping operations to their complexity costs.
    """

    def __init__(self, problem: 'SynthesisProblemZ3'):
        """
        Initialize the CandidateGenerator.

        Args:
            problem (SynthesisProblemZ3): The synthesis problem instance.
        """
        self.problem = problem
        self.config = problem.options
        self.min_const = self.config.synthesis_parameters.min_const
        self.max_const = self.config.synthesis_parameters.max_const
        self.operation_costs = self.config.synthesis_parameters.operation_costs

    @abstractmethod
    def generate_candidates(self) -> List[Tuple[z3.ExprRef, str]]:
        """
        Generate candidate functions.

        This method should be implemented by subclasses to generate a list of
        candidate functions for the synthesis problem.

        Returns:
            List[Tuple[z3.ExprRef, str]]: A list of tuples, where each tuple contains
            a Z3 expression representing a candidate function and its string representation.
        """
        pass

    @abstractmethod
    def prune_candidates(self, candidates: List[Tuple[z3.ExprRef, str]]) -> List[Tuple[z3.ExprRef, str]]:
        """
        Prune candidate functions.

        This method should be implemented by subclasses to filter out unsuitable
        candidates from the given list.

        Args:
            candidates (List[Tuple[z3.ExprRef, str]]): List of candidate functions to prune.

        Returns:
            List[Tuple[z3.ExprRef, str]]: A filtered list of candidate functions.
        """
        pass

    def op_complexity(self, op: str) -> int:
        """
        Get the complexity cost of an operation.

        Args:
            op (str): The operation name.

        Returns:
            int: The complexity cost of the operation, or 0 if not defined.
        """
        return self.operation_costs.get(op, 0)

    def get_arg_sorts(self, func_name: str) -> List[z3.SortRef]:
        """
        Get the argument sorts (types) for a given function.

        Args:
            func_name (str): The name of the function.

        Returns:
            List[z3.SortRef]: A list of Z3 sorts representing the argument types.
        """
        func = self.problem.context.z3_synth_functions[func_name]
        return [func.domain(i) for i in range(func.arity())]

    @staticmethod
    def create_candidate_function(candidate_expr: z3.ExprRef, arg_sorts: List[z3.SortRef]) -> z3.ExprRef:
        """
        Create a candidate function from an expression and argument sorts.

        This method creates a Z3 function by substituting variables in the candidate
        expression with fresh variables of the appropriate sorts.

        Args:
            candidate_expr (z3.ExprRef): The candidate expression.
            arg_sorts (List[z3.SortRef]): List of argument sorts for the function.

        Returns:
            z3.ExprRef: A Z3 expression representing the candidate function.
        """
        GlobalCancellationToken.check_cancellation()
        args = [z3.Var(i, sort) for i, sort in enumerate(arg_sorts)]
        return z3.substitute(candidate_expr, [(arg, z3.Var(i, arg.sort())) for i, arg in enumerate(args)])