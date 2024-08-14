import os
import sys
import threading
from dataclasses import asdict
from typing import Optional

from src.cegis.cvc5.synthesis_problem_cvc5 import SynthesisProblemCVC5
from src.cegis.z3.synthesis_strategy.partial_satisfaction_bottom_up import PartialSatisfactionBottomUp
from src.utilities.cancellation_token import CancellationError, GlobalCancellationToken
from src.utilities.config_manager import ConfigManager
from src.cegis.z3.synthesis_strategy.fast_enumerative_synth import FastEnumerativeSynthesis
from src.cegis.z3.synthesis_strategy.random_search_bottom_up import RandomSearchStrategyBottomUp
from src.cegis.z3.synthesis_strategy.random_search_top_down import RandomSearchStrategyTopDown
from src.cegis.z3.synthesis_problem_z3 import SynthesisProblemZ3
from src.utilities.options import Options

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# so we can log metrics even when it times out
problem_global: Optional[SynthesisProblemZ3 | SynthesisProblemCVC5] = None


def run_synthesis(config: Options) -> tuple[bool, Optional[str]]:
    """
    Run the synthesis process based on the provided configuration.

    This function sets up the synthesis problem, chooses the appropriate strategy,
    and executes the CEGIS (Counterexample-Guided Inductive Synthesis) loop.

    Args:
        config (Options): Configuration options for the synthesis process.

    Returns:
        Tuple[bool, Optional[str]]: A tuple containing a boolean indicating whether a solution was found,
        and a string representation of the solution candidates (or None if no solution was found).

    Raises:
        ValueError: If an unsupported solver or synthesis strategy is specified in the configuration.
        CancellationError: If the synthesis process is cancelled due to timing out.

    Example:
        >>> config = ConfigManager.get_config()
        >>> result, candidates = run_synthesis(config)
        >>> if result:
        ...     print(f"Solution found: {candidates}")
        ... else:
        ...     print("No solution found")

    Note:
        This function uses a global variable `problem_global` to store the problem instance,
        allowing for metric logging even if the synthesis process times out.
    """
    global problem_global
    with GlobalCancellationToken.cancellable():
        if config.input_source == 'stdin':
            problem_input = sys.stdin.read()
        else:
            with open(config.input_source, 'r') as file:
                problem_input = file.read()

        GlobalCancellationToken.check_cancellation()

        if config.solver.name.lower() == "z3":
            problem = SynthesisProblemZ3(problem_input, config)
        elif config.solver.name.lower() == "cvc5":
            problem = SynthesisProblemCVC5(problem_input, config)
        else:
            raise ValueError(f"Unsupported solver: {config.solver.name}")

        problem.logger.info(problem.info_smt())
        problem_global = problem

        GlobalCancellationToken.check_cancellation()

        if config.synthesis_parameters.strategy == 'fast_enumerative':
            strategy = FastEnumerativeSynthesis(problem)
        elif config.synthesis_parameters.strategy == 'random_enumerative':
            if config.synthesis_parameters.candidate_generation == 'top_down':
                strategy = RandomSearchStrategyTopDown(problem)
            else:
                strategy = RandomSearchStrategyBottomUp(problem)
        elif config.synthesis_parameters.strategy == 'partial':
            if config.synthesis_parameters.candidate_generation == 'top_down':
                strategy = RandomSearchStrategyTopDown(problem)
            else:
                strategy = PartialSatisfactionBottomUp(problem)
        else:
            ConfigManager.logger.error(f"Unknown synthesis strategy: {config.synthesis_parameters.strategy}")
            raise ValueError(f"Unknown synthesis strategy: {config.synthesis_parameters.strategy}")
        result, candidates = strategy.execute_cegis()
        return result, candidates


def main() -> None:
    """
    This function sets up the synthesis configuration, runs the synthesis in a separate thread
    with a timeout, logs metrics, and handles various outcomes of the synthesis process.

    Note:
        The function uses a global variable `problem_global` to access the problem instance
        for metric logging, even if the synthesis process times out.
    """
    global problem_global
    config: Options = ConfigManager.get_config()
    ConfigManager.logger.info(asdict(config))

    result = None
    candidates = None
    exit_code = 0

    def thread_target():
        nonlocal result, candidates
        try:
            result, candidates = run_synthesis(config)
        except CancellationError:
            print(f"Synthesis was cancelled after {config.synthesis_parameters.timeout} seconds")
        except Exception as e:
            print(f"An error occurred during synthesis: {str(e)}")
            nonlocal exit_code
            exit_code = 1

    thread = threading.Thread(target=thread_target)
    thread.start()
    thread.join(timeout=config.synthesis_parameters.timeout)

    if thread.is_alive():
        GlobalCancellationToken().cancel()
        thread.join()
        print(f"Synthesis timed out after {config.synthesis_parameters.timeout} seconds")
        exit_code = -6

    if problem_global is not None:
        try:
            metrics_summary = problem_global.metrics.get_summary()
            print("Metrics summary:", metrics_summary)
        except Exception as e:
            print(f"Error logging metrics: {str(e)}")

    if result:
        print(candidates)
        exit_code = 0
    elif exit_code == 0:
        print("No satisfying candidates found.")
        exit_code = 2

    sys.exit(exit_code)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)
