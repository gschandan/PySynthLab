from itertools import product
from src.cegis.z3.candidate_generator.fast_enumerative_candidate_generator import FastEnumerativeCandidateGenerator
from src.cegis.z3.synthesis_problem import SynthesisProblem
from src.cegis.z3.synthesis_strategy.synthesis_strategy import SynthesisStrategy


class FastEnumerativeSynthesis(SynthesisStrategy):
    """
    A synthesis strategy that uses fast enumerative synthesis.

    This class implements a synthesis strategy based on the fast enumerative synthesis approach,
    which systematically explores the space of possible programs in a systematic manner.

    Reynolds et al., "CVC4SY: Smart and Fast Term Enumeration for Syntax-Guided Synthesis" (CAV 2019)
    http://homepage.divms.uiowa.edu/~ajreynol/cav19b.pdf

    Attributes:
        problem (SynthesisProblem): The synthesis problem to be solved.
        candidate_generator (FastEnumerativeCandidateGenerator): The generator used to produce candidate solutions.

    Example:
        .. code-block:: python

            from src.cegis.z3.synthesis_problem import SynthesisProblem
            from src.cegis.z3.synthesis_strategy.fast_enumerative_synth import FastEnumerativeSynthesis

            # Define the synthesis problem
            problem = SynthesisProblem(...)

            # Create the FastEnumerativeSynthesis strategy
            strategy = FastEnumerativeSynthesis(problem)

            # Execute the CEGIS loop
            strategy.execute_cegis()

            # Check if a solution was found
            if strategy.solution_found:
                print("Solution found!")
            else:
                print("No solution found within the given constraints.")

    Note:
        This strategy is particularly effective for problems where the space of possible
        programs can be systematically enumerated and pruned efficiently. It can be extended to leverage
        divide-and-conquer techniques to scale to larger synthesis tasks.
    """

    def __init__(self, problem: SynthesisProblem):
        """
        Initialize the FastEnumerativeSynthesis strategy.

        Args:
            problem (SynthesisProblem): The synthesis problem to be solved.
        """
        super().__init__(problem)
        self.problem = problem
        self.candidate_generator = FastEnumerativeCandidateGenerator(problem)

    def execute_cegis(self) -> None:
        """
        Execute the CEGIS (Counterexample-Guided Inductive Synthesis) loop using fast enumerative synthesis.

        This method implements two different approaches based on the number of synthesis functions:

        1. For a single synthesis function, it generates and tests candidates up to the configurable maximum number of iterations.
        2. For multiple synthesis functions, it generates candidates at increasing depths and tests combinations of candidates.

        The method stops when a solution is found or when the maximum number of iterations or depth is reached.

        Note:
            The specific behavior and termination conditions can be customized through
            the synthesis_parameters in the configuration file or command line arguments.

        Raises:
            ValueError: If the number of candidate functions doesn't match the number of synthesis functions.
        """
        max_iterations = self.problem.options.synthesis_parameters.max_iterations
        max_depth = self.problem.options.synthesis_parameters.max_depth
        synth_func_names = list(self.problem.context.z3_synth_functions.keys())

        if len(synth_func_names) <= 1:
            self._execute_single_function_cegis(max_iterations)
            return
        else:
            self._execute_multi_function_cegis(max_iterations, max_depth, synth_func_names)
            return

    def _execute_single_function_cegis(self, max_iterations: int) -> None:
        """
        Execute CEGIS for a single synthesis function.

        Args:
            max_iterations (int): The maximum number of iterations to perform.
        """
        for iteration in range(max_iterations):
            SynthesisProblem.logger.info(f"Iteration {iteration + 1}/{max_iterations}")
            candidates = self.candidate_generator.generate_candidates()
            if not candidates:
                continue
            for candidate, func_name in candidates:
                SynthesisProblem.logger.info(f"Testing candidate: {func_name}: {str(candidate)}")
                if self.test_candidates([func_name], [candidate]):
                    SynthesisProblem.logger.info(f"Found satisfying candidate!")
                    SynthesisProblem.logger.info(f"{func_name}: {candidate}")
                    self.set_solution_found()
                    return

        SynthesisProblem.logger.info(f"No solution found after {max_iterations} iterations")

    def _execute_multi_function_cegis(self, max_iterations: int, max_depth: int, synth_func_names: list[str]) -> None:
        """
        Execute CEGIS for multiple synthesis functions.

        Args:
            max_depth (int): The maximum depth to explore in the candidate space.
            max_iterations (int): The maximum number of iterations to perform.
        """
        iteration = 0
        for depth in range(max_depth + 1):
            SynthesisProblem.logger.info(f"Depth {depth}/{max_depth}")

            all_candidates = {func_name: [] for func_name in synth_func_names}
            for candidate, func_name in self.candidate_generator.generate_candidates_at_depth(depth):
                all_candidates[func_name].append(candidate)
            for func_name, candidates in all_candidates.items():
                SynthesisProblem.logger.debug(f"Generated {len(candidates)} candidates for {func_name} at depth {depth}")

            if any(not candidates for candidates in all_candidates.values()):
                SynthesisProblem.logger.warning(f"Missing candidates for some functions at depth {depth}")
                return

            for candidate_combination in product(*(all_candidates[func_name] for func_name in synth_func_names)):
                func_strs = synth_func_names
                candidate_functions = list(candidate_combination)

                SynthesisProblem.logger.info(
                    f"Testing candidates: {'; '.join([f'{func}: {cand}' for func, cand in zip(func_strs, candidate_functions)])}")

                if self.test_candidates(func_strs, candidate_functions):
                    SynthesisProblem.logger.info(f"Found satisfying candidates!")
                    for func_name, candidate in zip(func_strs, candidate_functions):
                        SynthesisProblem.logger.info(f"{func_name}: {candidate}")
                    self.set_solution_found()
                    return
            iteration += 1
            if iteration >= max_iterations:
                SynthesisProblem.logger.info(
                    f"No satisfying candidates found within {max_iterations} iterations.")
                return

        SynthesisProblem.logger.info(f"No solution found up to depth {max_depth}")
