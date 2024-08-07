from abc import ABC, abstractmethod
from typing import List, Tuple
from z3 import *

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

    def __init__(self, problem: SynthesisProblemZ3):
        """
        Initialize the CandidateGenerator.

        Args:
            problem (SynthesisProblemZ3): The synthesis problem instance.
        """
        self.problem = problem
        self.metrics = problem.metrics
        self.config = problem.options
        self.min_const = self.config.synthesis_parameters.min_const
        self.max_const = self.config.synthesis_parameters.max_const
        self.operation_costs = self.config.synthesis_parameters.operation_costs

        self.partial_solver = Solver()
        self.partial_solver.set('smt.macro_finder', True)
        self.partial_solver.set('timeout', self.problem.options.solver.timeout)
        self.partial_solver.set('random_seed', self.problem.options.synthesis_parameters.random_seed)

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

    def check_partial_satisfaction(self, candidate: ExprRef, func_name: str) -> float:
        """
        Check partial satisfaction of constraints by the candidate solution.

        This method checks each constraint individually and returns the fraction satisfied.

        Args:
            candidate (ExprRef): The candidate solution to evaluate.
            func_name (str): The name of the function being synthesized.

        Returns:
            float: The fraction of constraints satisfied by the candidate.

        Example:
            score = generator.check_partial_satisfaction(candidate, 'f')
        """
        constraints = self.problem.context.z3_non_conjoined_constraints
        self.partial_solver.reset()
        satisfied_constraints = 0
        for constraint in constraints:
            GlobalCancellationToken.check_cancellation()
            self.partial_solver.push()
            if is_implies(constraint):
                antecedent, consequent = constraint.arg(0), constraint.arg(1)
                negated_constraint = And(antecedent, Not(consequent))
            else:
                negated_constraint = Not(constraint)
            substituted_constraint = self.problem.substitute_constraints(
                [negated_constraint],
                [self.problem.context.z3_synth_functions[func_name]],
                [candidate])
            self.partial_solver.add(substituted_constraint)
            if self.partial_solver.check() == unsat:
                satisfied_constraints += 1
            self.partial_solver.pop()
        return satisfied_constraints / len(constraints)

    def quantitative_satisfaction(self, candidate: ExprRef, func_name: str) -> float:
        """
        Evaluate the candidate using quantitative satisfaction.

        This method measures how close the candidate is to satisfying each constraint,
        providing a more nuanced score than binary satisfaction.

        Args:
            candidate (ExprRef): The candidate solution to evaluate.
            func_name (str): The name of the function being synthesized.

        Returns:
            float: A score between 0 and 1, where 1 indicates full satisfaction.

        Example:
            score = generator.quantitative_satisfaction(candidate, 'f')
        """
        self.partial_solver.reset()
        total_diff = 0.0
        for constraint in self.problem.context.z3_non_conjoined_constraints:
            GlobalCancellationToken.check_cancellation()
            if is_implies(constraint):
                antecedent, consequent = constraint.arg(0), constraint.arg(1)
                negated_constraint = And(antecedent, Not(consequent))
            else:
                negated_constraint = Not(constraint)
            substituted_constraint = self.problem.substitute_constraints(
                [negated_constraint],
                [self.problem.context.z3_synth_functions[func_name]],
                [candidate])[0]
            if is_bool(substituted_constraint):
                self.partial_solver.push()
                self.partial_solver.add(substituted_constraint)
                if self.partial_solver.check() == unsat:
                    total_diff += 0
                else:
                    total_diff += 1
                self.partial_solver.pop()
            elif is_arith(substituted_constraint):
                diff = Abs(substituted_constraint)
                self.partial_solver.push()
                self.partial_solver.add(diff >= 0)
                if self.partial_solver.check() == unsat:
                    diff_value = self.partial_solver.model().eval(diff)
                    if is_rational_value(diff_value):
                        total_diff += diff_value.as_fraction()
                    elif is_int_value(diff_value):
                        total_diff += float(diff_value.as_long())
                    else:
                        total_diff += 1
                self.partial_solver.pop()
            else:
                total_diff += 1
        return 1.0 / (1.0 + total_diff)

    def unsat_core_analysis(self, candidate: ExprRef, func_name: str) -> float:
        """
        Evaluate the candidate using unsat core analysis.

        This method uses Z3's unsat core providing a score and insight into how close the candidate
        is to satisfying all constraints.

        Args:
            candidate (ExprRef): The candidate solution to evaluate.
            func_name (str): The name of the function being synthesized.

        Returns:
            float: The fraction of constraints not in the unsat core (i.e., satisfied).

        Example:
            score = generator.unsat_core_analysis(candidate, 'f')
        """
        self.partial_solver.reset()
        negated = self.problem.substitute_constraints(
            self.problem.context.z3_negated_constraints,
            [self.problem.context.z3_synth_functions[func_name]],
            [candidate])
        self.partial_solver.assert_exprs(negated)
        if self.partial_solver.check() == unsat:
            return 1.0

        self.partial_solver.reset()
        self.partial_solver.set(unsat_core=True)

        substituted_constraints = self.problem.substitute_constraints(
            self.problem.context.z3_non_conjoined_constraints,
            [self.problem.context.z3_synth_functions[func_name]],
            [candidate])
        for i, constraint in enumerate(substituted_constraints):
            self.partial_solver.assert_and_track(Not(constraint), Bool(f'c_{i}'))

        result = self.partial_solver.check()

        if result == unsat:
            unsat_core = self.partial_solver.unsat_core()
            satisfied_constraints = len(substituted_constraints) - len(unsat_core)
        else:
            satisfied_constraints = 0

        score = satisfied_constraints / len(substituted_constraints)

        self.partial_solver.set(unsat_core=False)
        return score

    def fuzzy_satisfaction(self, candidate: ExprRef, func_name: str) -> float:
        """
        Evaluate the candidate using fuzzy satisfaction.

        This method checks each constraint individually but allows for partial
        satisfaction, providing a more gradual measure of constraint satisfaction.

        Args:
            candidate (ExprRef): The candidate solution to evaluate.
            func_name (str): The name of the function being synthesized.

        Returns:
            float: A score between 0 and 1, where 1 indicates full satisfaction.

        Example:
            score = generator.fuzzy_satisfaction(candidate, 'f')
        """
        self.partial_solver.reset()
        all_satisfied = True
        num_satisfied = 0

        for constraint in self.problem.context.z3_non_conjoined_constraints:
            GlobalCancellationToken.check_cancellation()

            if is_implies(constraint):
                antecedent, consequent = constraint.arg(0), constraint.arg(1)
                negated_constraint = And(antecedent, Not(consequent))
            else:
                negated_constraint = Not(constraint)

            substituted_constraint = self.problem.substitute_constraints(
                [negated_constraint],
                [self.problem.context.z3_synth_functions[func_name]],
                [candidate])
            self.partial_solver.push()
            self.partial_solver.add(substituted_constraint)
            if self.partial_solver.check() == unsat:
                num_satisfied += 1
            else:
                all_satisfied = False
            self.partial_solver.pop()

        if all_satisfied:
            return 1.0
        else:
            return num_satisfied / len(self.problem.context.z3_non_conjoined_constraints)