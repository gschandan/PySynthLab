from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
from z3 import *
from z3 import ExprRef

from src.cegis.z3.candidate_generator.candidate_generator_base import CandidateGenerator
from src.cegis.z3.candidate_generator.random_candidate_generator import RandomCandidateGenerator
from src.cegis.z3.synthesis_problem_z3 import SynthesisProblemZ3


class SynthesisStrategy(ABC):
    """
    An abstract base class for implementing various synthesis strategies.

    This class provides a framework for implementing different approaches to program synthesis,
    particularly in the context of Counterexample-Guided Inductive Synthesis (CEGIS).

    Attributes:
        problem (SynthesisProblemZ3): The synthesis problem to be solved.
        solution_found (bool): Flag indicating whether a solution has been found.
        candidate_generator (CandidateGenerator): The generator used to produce candidate solutions.

    Example:
        .. code-block:: python

            class MyCustomStrategy(SynthesisStrategy):
                def execute_cegis(self):
                    max_iterations = 100
                    for iteration in range(max_iterations):
                        candidates = self.generate_candidates()
                        for candidate, func_name in candidates:
                            if self.test_candidates([func_name], [candidate]):
                                self.set_solution_found()
                                return True, candidate
                    return False, "No solution found within max iterations."

    Usage:
        .. code-block:: python

            problem = SynthesisProblem(...)
            strategy = MyCustomStrategy(problem)
            strategy.execute_cegis()
    """

    def __init__(self, problem: 'SynthesisProblemZ3', candidate_generator: CandidateGenerator = None):
        """
        Initialize the SynthesisStrategy.

        Args:
            problem (SynthesisProblemZ3): The synthesis problem to be solved.
            candidate_generator (CandidateGenerator, optional): A custom candidate generator. If None, a default RandomCandidateGenerator will be used.
        """
        self.problem = problem
        self.solution_found = False
        if candidate_generator is None:
            self.candidate_generator = RandomCandidateGenerator(problem)
        else:
            self.candidate_generator = candidate_generator

    def generate_candidates(self) -> list[tuple[ExprRef, str]]:
        """
        Generate candidate solutions using the candidate generator.

        Returns:
            list[tuple[ExprRef, str]]: A list of tuples, each containing a candidate solution
            (as a Z3 expression) and its string representation.
        """

        return self.candidate_generator.generate_candidates()

    @abstractmethod
    def execute_cegis(self) -> tuple[bool, str]:
        """
        Execute the CEGIS loop.

        This method should implement the core logic of the synthesis strategy.
        It typically involves generating candidates, testing them, and refining
        the search based on counterexamples until a solution is found or
        a termination condition is met.

        This is an abstract method that must be implemented by subclasses.
        """
        pass

    def set_solution_found(self) -> None:
        """
        Mark that a solution has been found.

        This method sets the solution_found flag to True, indicating that
        the synthesis process has successfully found a solution.
        """
        self.solution_found = True

    def set_candidate_generator(self, generator: CandidateGenerator):
        """
        Set a new candidate generator.

        This method allows changing the candidate generator during the synthesis process,
        which can be useful for implementing adaptive or other custom strategies.

        Args:
            generator (CandidateGenerator): The new candidate generator to use.
        """
        self.candidate_generator = generator

    def test_candidates(self, func_strs: List[str], candidate_functions: List[z3.ExprRef]) -> bool:
        """
        Test a set of candidate functions against the synthesis problem.

        This method checks if the given candidate functions satisfy all constraints
        and pass stored counterexamples.

        Args:
           func_strs (List[str]): List of function names corresponding to the candidates.
           candidate_functions (List[z3.ExprRef]): List of candidate functions as Z3 expressions.

        Returns:
           bool: True if the candidates satisfy all constraints, False otherwise.

        Raises:
           ValueError: If the number of candidate functions doesn't match the number of synthesis functions.
       """

        synth_func_names = list(self.problem.context.z3_synth_functions.keys())
        self.problem.logger.debug(f" candidate_functions {candidate_functions}")

        if len(func_strs) != len(synth_func_names):
            raise ValueError("Number of candidate functions doesn't match number of synthesis functions")

        for func, candidate, synth_func_name in zip(func_strs, candidate_functions, synth_func_names):
            if not self.check_counterexample(synth_func_name, candidate):
                return False
        self.problem.logger.debug("All individual counterexample checks passed")

        candidates = list(zip(candidate_functions, synth_func_names))
        new_counterexamples = self.generate_counterexample(candidates)
        if new_counterexamples is not None:
            for func_name, ce in new_counterexamples.items():
                self.problem.logger.info(f"New counterexample found for {func_name}: {ce} Candidates: {candidates}")
            return False
        self.problem.logger.debug("No new counterexamples generated")

        if not self.verify_candidates(candidate_functions):
            self.problem.logger.info(
                f"Verification failed for guess {'; '.join(func_strs)}. Candidates violate constraints.")
            return False

        self.problem.logger.info(f"No counterexample found! Guess(es) should be correct: {'; '.join(func_strs)}.")
        return True

    def check_counterexample(self, func_name: str, candidate: z3.ExprRef) -> bool:
        """
        Check if a candidate satisfies all stored counterexamples for a given function.

        Args:
            func_name (str): Name of the function being synthesized.
            candidate (z3.ExprRef): The candidate function to check.

        Returns:
            bool: True if the candidate satisfies all counterexamples, False otherwise.
        """
        for stored_func_name, ce in self.problem.context.counterexamples:
            if stored_func_name == func_name:
                synth_func = self.problem.context.z3_synth_functions[func_name]

                substituted_constraints = self.problem.substitute_constraints(
                    self.problem.context.z3_constraints,
                    [synth_func],
                    [candidate]
                )

                solver = z3.Solver()

                for var, value in ce.items():
                    solver.add(var == value)

                solver.add(substituted_constraints)

                if solver.check() == z3.unsat:
                    return False
        return True

    def generate_counterexample(self, candidates: List[Tuple[z3.ExprRef, str]]) -> Dict[str, Dict[str, Any]] | None:
        """
        Find a potential counterexample for a set of candidates, if one can be found.

        This method attempts to find input values that violate the problem constraints
        for the given candidate functions.

        Args:
            candidates (List[Tuple[z3.ExprRef, str]]): List of tuples containing candidate functions and their names.

        Returns:
            Dict[str, Dict[str, Any]] | None: A dictionary of counterexamples for each function,
            or None if no counterexample is found.
        """
        self.problem.context.enumerator_solver.reset()
        substituted_neg_constraints = self.problem.substitute_constraints(
            self.problem.context.z3_negated_constraints,
            list(self.problem.context.z3_synth_functions.values()),
            [candidate for candidate, _ in candidates])
        self.problem.context.enumerator_solver.add(substituted_neg_constraints)
        self.problem.logger.debug(f"Negated constraints: {self.problem.context.z3_negated_constraints}")
        self.problem.logger.debug(f"Substituted negated constraints: {substituted_neg_constraints}")
        result = self.problem.context.enumerator_solver.check()
        self.problem.logger.debug(f"generate_counterexample solver check result: {result}")

        if result == z3.sat:
            model = self.problem.context.enumerator_solver.model()
            counterexamples = {}

            for (candidate, synth_func_name) in candidates:
                variable_mapping = self.problem.context.variable_mapping_dict[synth_func_name]
                args = list(variable_mapping.values())
                ce = {arg: model.eval(arg, model_completion=True) for arg in args}
                self.problem.logger.info(f"Counterexample for {synth_func_name}: {ce}")
                counterexamples[synth_func_name] = ce
                self.problem.context.counterexamples.append((synth_func_name, ce))

            return counterexamples
        else:
            self.problem.logger.info(f"No counterexample found for candidates {candidates}")
            return None

    def verify_candidates(self, candidates: List[z3.ExprRef]) -> bool:
        """
        Verify if a set of candidates satisfies all constraints.

        This method performs a final verification step to ensure that the candidate functions
        satisfy all constraints of the synthesis problem.

        Args:
           candidates (List[z3.ExprRef]): List of candidate functions to verify.

        Returns:
           bool: True if the candidates satisfy all constraints, False otherwise.
       """

        self.problem.context.verification_solver.reset()
        substituted_constraints = self.problem.substitute_constraints(
            self.problem.context.z3_constraints,
            list(self.problem.context.z3_synth_functions.values()),
            candidates)
        self.problem.context.verification_solver.add(substituted_constraints)

        return self.problem.context.verification_solver.check() == z3.sat
