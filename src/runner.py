import os
import sys
import threading
from dataclasses import asdict
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


def run_synthesis(config: Options):
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

        GlobalCancellationToken.check_cancellation()

        if config.synthesis_parameters.strategy == 'fast_enumerative':
            strategy = FastEnumerativeSynthesis(problem)
        elif config.synthesis_parameters.strategy == 'random_enumerative':
            if config.synthesis_parameters.candidate_generation == 'top_down':
                strategy = RandomSearchStrategyTopDown(problem)
            else:
                strategy = RandomSearchStrategyBottomUp(problem)
        elif config.synthesis_parameters.strategy == 'partial':
            strategy = PartialSatisfactionBottomUp(problem)
        else:
            ConfigManager.logger.error(f"Unknown synthesis strategy: {config.synthesis_parameters.strategy}")
            raise ValueError(f"Unknown synthesis strategy: {config.synthesis_parameters.strategy}")

        result, candidates = strategy.execute_cegis()
        return result, candidates


def main() -> None:
    try:
        config: Options = ConfigManager.get_config()
        ConfigManager.logger.info(asdict(config))

        def thread_target():
            try:
                result, candidates = run_synthesis(config)
                if result:
                    print(candidates)
                else:
                    print("No satisfying candidates found.")
            except CancellationError:
                print(f"Synthesis was cancelled after {config.synthesis_parameters.timeout} seconds")

        thread = threading.Thread(target=thread_target)
        thread.start()
        thread.join(timeout=config.synthesis_parameters.timeout)

        if thread.is_alive():
            GlobalCancellationToken().cancel()
            thread.join()
            print(f"Synthesis timed out after {config.synthesis_parameters.timeout} seconds")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == '__main__':
    main()
