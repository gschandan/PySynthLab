import time

from src.cegis.z3.candidate_generator.top_down_enumerative_generator import TopDownCandidateGenerator
from src.cegis.z3.candidate_generator.weighted_top_down_enumerative_generator import WeightedTopDownCandidateGenerator
from src.cegis.z3.synthesis_problem_z3 import SynthesisProblemZ3
from src.cegis.z3.synthesis_strategy.synthesis_strategy import SynthesisStrategy


class RandomSearchStrategyTopDown(SynthesisStrategy):
    """
    A synthesis strategy that uses a top-down random search approach.

    This class implements a synthesis strategy based on a top-down random search. It
    starts with complex expressions and gradually simplifies them, exploring the
    solution space in a top-down manner. This approach can be effective for problems
    where the general structure of the solution is known, but the specific details
    need to be refined.

    The strategy can use either a regular top-down candidate generator or a weighted
    top-down candidate generator, depending on the configuration.

    Attributes:
        problem (SynthesisProblemZ3): The synthesis problem to be solved.
        candidate_generator (Union[TopDownCandidateGenerator, WeightedTopDownCandidateGenerator]): 
            The generator used to produce candidate solutions.

    Example:
        .. code-block:: python

            from src.cegis.z3.synthesis_problem_z3 import SynthesisProblemZ3
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
        the solution space is well-defined.

        The use of a weighted generator can be controlled through the
        `use_weighted_generator` option in the synthesis parameters.
    """

    def __init__(self, problem: SynthesisProblemZ3):
        """
        Initialize the RandomSearchStrategyTopDown strategy.

        Args:
            problem (SynthesisProblemZ3): The synthesis problem to be solved.
        """
        super().__init__(problem)
        self.problem = problem
        if problem.options.synthesis_parameters.use_weighted_generator:
            self.candidate_generator = WeightedTopDownCandidateGenerator(problem)
        else:
            self.candidate_generator = TopDownCandidateGenerator(problem)

        if problem.options.synthesis_parameters.custom_grammar:
            self.candidate_generator.grammar = problem.options.synthesis_parameters.custom_grammar
            self.metrics.grammar = sum(len(rules) for rules in self.candidate_generator.grammar.values())


    def execute_cegis(self) -> tuple[bool, str]:
        """
        Execute the CEGIS loop using a top-down random search strategy.

        This method generates candidates using a top-down approach, and tests them against
        the synthesis problem. The search continues for a specified number of iterations or until a
        solution is found.

        The method uses the following parameter from the synthesis problem options:
        - max_iterations: The maximum number of iterations to perform.

        Note:
            The specific behavior and termination conditions can be customized through
            the synthesis parameters in the problem definition.
        """
        start_time = time.time()
        max_iterations = self.problem.options.synthesis_parameters.max_iterations

        for iteration in range(max_iterations):
            self.metrics.iterations += 1

            candidates = self.candidate_generator.generate_candidates()
            self.metrics.candidates_generated += len(candidates)

            if all(candidate is None for candidate in candidates):
                continue

            pruned_candidates = self.candidate_generator.prune_candidates(candidates)
            self.metrics.candidates_pruned += len(candidates) - len(pruned_candidates)

            self.problem.logger.info(f"Iteration {iteration + 1}/{max_iterations}:\n")
            func_strs = [f"{func_name}: {candidate}" for candidate, func_name in pruned_candidates]
            candidate_functions = [candidate for candidate, _ in pruned_candidates]

            if self.test_candidates(func_strs, candidate_functions):
                self.metrics.time_spent = time.time() - start_time
                self.metrics.candidates_generated += len(candidates)

                self.problem.logger.info(f"Found satisfying candidates!")
                valid_candidates = ''
                for candidate, func_name in pruned_candidates:
                    self.problem.logger.info(f"{func_name}: {candidate}")
                    valid_candidates += f"{func_name}: {candidate}\n"
                self.set_solution_found()
                return True, valid_candidates

        self.problem.logger.info(f"No satisfying candidates found within {max_iterations} iterations.")
        return False, f"No satisfying candidates found within {max_iterations} iterations."
