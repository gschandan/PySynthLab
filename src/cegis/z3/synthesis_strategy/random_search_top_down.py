from src.cegis.z3.candidate_generator.top_down_enumerative_generator import TopDownCandidateGenerator
from src.cegis.z3.candidate_generator.weighted_top_down_enumerative_generator import WeightedTopDownCandidateGenerator
from src.cegis.z3.synthesis_problem import SynthesisProblem
from src.cegis.z3.synthesis_strategy.synthesis_strategy import SynthesisStrategy


class RandomSearchStrategyTopDown(SynthesisStrategy):
    """
    A synthesis strategy that uses a top-down random search approach.

    This class implements a synthesis strategy based on a top-down random search. It
    starts with complex expressions and gradually simplifies them, exploring the
    solution space in a top-down manner. This approach can be effective for problems
    where the general structure of the solution is known, but the specific details
    need to be refined.

    Attributes:
        problem (SynthesisProblem): The synthesis problem to be solved.
        candidate_generator (TopDownCandidateGenerator): The generator used to produce candidate solutions.

    Example:
        .. code-block:: python

            from src.cegis.z3.synthesis_problem import SynthesisProblem
            from src.cegis.z3.synthesis_strategy.random_search_top_down import RandomSearchStrategyTopDown

            # Define the synthesis problem
            problem = SynthesisProblem(...)

            # Create the RandomSearchStrategyTopDown strategy
            strategy = RandomSearchStrategyTopDown(problem)

            # Execute the CEGIS loop
            strategy.execute_cegis()

            # Check if a solution was found
            if strategy.solution_found:
                print("Solution found!")
            else:
                print("No solution found within the given constraints.")

    Note:
        This strategy is particularly useful when you have an idea of the overall
        structure of the solution and want to refine it. It can be more efficient
        than bottom-up approaches for certain types of problems, especially when
        the solution space is well-structured.
    """

    def __init__(self, problem: SynthesisProblem):
        """
        Initialize the RandomSearchStrategyTopDown strategy.

        Args:
            problem (SynthesisProblem): The synthesis problem to be solved.
        """
        super().__init__(problem)
        self.problem = problem
        # self.candidate_generator = TopDownCandidateGenerator(problem)
        self.candidate_generator = WeightedTopDownCandidateGenerator(problem)

    def execute_cegis(self) -> None:
        """
        Execute the CEGIS (Counterexample-Guided Inductive Synthesis) loop using a top-down random search strategy.

        This method generates candidates using a top-down approach, prunes them, and tests them against
        the synthesis problem. The search continues for a specified number of iterations or until a
        solution is found.

        The method uses the following parameter from the synthesis problem options:
        - max_iterations: The maximum number of iterations to perform.

        Note:
            The specific behavior and termination conditions can be customized through
            the synthesis parameters in the problem definition.
        """

        max_iterations = self.problem.options.synthesis_parameters.max_iterations

        for iteration in range(max_iterations):
            candidates = self.candidate_generator.generate_candidates()
            if all(candidate is None for candidate in candidates):
                continue
            pruned_candidates = self.candidate_generator.prune_candidates(candidates)

            SynthesisProblem.logger.info(f"Iteration {iteration + 1}/{max_iterations}:\n")
            func_strs = [f"{func_name}: {candidate}" for candidate, func_name in pruned_candidates]
            candidate_functions = [candidate for candidate, _ in pruned_candidates]

            if self.test_candidates(func_strs, candidate_functions):
                SynthesisProblem.logger.info(f"Found satisfying candidates!")
                for candidate, func_name in pruned_candidates:
                    SynthesisProblem.logger.info(f"{func_name}: {candidate}")
                self.set_solution_found()
                return

        SynthesisProblem.logger.info("No satisfying candidates found.")
