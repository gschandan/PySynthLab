from src.cegis.z3.synthesis_problem_z3 import SynthesisProblemZ3
from src.cegis.z3.synthesis_strategy.synthesis_strategy import SynthesisStrategy


class RandomSearchStrategyBottomUp(SynthesisStrategy):

    """
    A synthesis strategy that uses a bottom-up random search approach.

    This class implements a synthesis strategy based on a bottom-up random search. It
    incrementally builds candidate solutions by starting with simple expressions and constants, and
    gradually increasing their complexity. This approach can be effective for problems
    where the solution space is large and a systematic search might be too time-consuming.

    Additional control is provided via adjusting the probabilities and weighting of operation selection in the
    candidate generator.

    Attributes:
        problem (SynthesisProblemZ3): The synthesis problem to be solved.
        min_const (int): The minimum constant value to consider in candidate generation.
        max_const (int): The maximum constant value to consider in candidate generation.

    Example:
        .. code-block:: python

            from src.cegis.z3.synthesis_problem_z3 import SynthesisProblemZ3
            from src.cegis.z3.synthesis_strategy.random_search_bottom_up import RandomSearchStrategyBottomUp

            # Define the synthesis problem
            problem = SynthesisProblem(...)

            # Create the RandomSearchStrategyBottomUp strategy
            strategy = RandomSearchStrategyBottomUp(problem)

            # Execute the CEGIS loop
            strategy.execute_cegis()

            # Check if a solution was found
            if strategy.solution_found:
                print("Solution found!")
            else:
                print("No solution found within the given constraints.")

    Note:
        This strategy is particularly useful when the structure of the solution is unknown
        and a more exploratory approach is needed. It can sometimes find solutions more
        quickly than exhaustive search strategies, especially for problems with a large
        search space.
    """

    def __init__(self, problem: SynthesisProblemZ3):
        """
        Initialize the RandomSearchStrategyBottomUp strategy.

        Args:
            problem (SynthesisProblemZ3): The synthesis problem to be solved.
        """

        super().__init__(problem)
        self.problem = problem
        self.min_const = problem.options.synthesis_parameters.min_const
        self.max_const = problem.options.synthesis_parameters.max_const

    def execute_cegis(self) -> None:
        """
        Execute the CEGIS (Counterexample-Guided Inductive Synthesis) loop using a bottom-up random search strategy.

        This method generates candidates by incrementally increasing depth and complexity. It prunes candidates
        and tests them against the synthesis problem. The search continues until a solution is found or the
        maximum number of iterations, depth, or complexity is reached.

        The method uses the following parameters from the synthesis problem options:
        - max_depth: The maximum depth of expressions to consider.
        - max_complexity: The maximum complexity of expressions to consider.
        - max_candidates_at_each_depth: The maximum number of candidates to generate and test at each depth.
        - max_iterations: The total maximum number of iterations to perform.

        Note:
            The specific behavior and termination conditions can be customized through
            the synthesis parameters in the problem definition.
        """

        max_depth = self.problem.options.synthesis_parameters.max_depth
        max_complexity = self.problem.options.synthesis_parameters.max_complexity
        max_candidates_per_depth = self.problem.options.synthesis_parameters.max_candidates_at_each_depth
        max_iterations = self.problem.options.synthesis_parameters.max_iterations
        iteration = 0
        for depth in range(1, max_depth + 1):
            for complexity in range(1, max_complexity + 1):
                for candidate_at_depth in range(max_candidates_per_depth):
                    candidates = self.candidate_generator.generate_candidates()
                    pruned_candidates = self.candidate_generator.prune_candidates(candidates)
                    self.problem.logger.info(f"Iteration {iteration + 1}/{max_iterations} depth: {depth}, complexity: {complexity}, candidate at depth: {candidate_at_depth + 1}/{max_candidates_per_depth}):\n")
                    func_strs = [f"{func_name}: {candidate}" for candidate, func_name in pruned_candidates]
                    candidate_functions = [candidate for candidate, _ in pruned_candidates]
                    if self.test_candidates(func_strs, candidate_functions):
                        self.problem.logger.info(f"Found satisfying candidates!")
                        for candidate, func_name in pruned_candidates:
                            self.problem.logger.info(f"{func_name}: {candidate}")
                        self.set_solution_found()
                        return
                    iteration += 1
                    if iteration >= max_iterations:
                        self.problem.logger.info(f"No satisfying candidates found within {max_iterations} iterations.")
                        return
        self.problem.logger.info("No satisfying candidates found.")
