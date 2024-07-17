import json
import sys
from dataclasses import asdict

from src.cegis.z3.config_manager import ConfigManager
from src.cegis.z3.fast_enumerative_synth import FastEnumerativeSynthesis
from src.cegis.z3.random_search_bottom_up import RandomSearchStrategyBottomUp
from src.cegis.z3.random_search_top_down import RandomSearchStrategyTopDown
from src.cegis.z3.synthesis_problem import SynthesisProblem


def main() -> None:
    config = ConfigManager.get_config()
    print(asdict(config))

    if config.input_source == 'stdin':
        problem_input = sys.stdin.read()
    else:
        with open(config.input_source, 'r') as file:
            problem_input = file.read()

    problem = SynthesisProblem(problem_input, config)
    print(problem.info_smt())

    if config.synthesis_parameters.strategy == 'fast_enumerative':
        strategy = FastEnumerativeSynthesis(problem)
    elif config.synthesis_parameters.strategy == 'random_enumerative':
        if config.synthesis_parameters.candidate_generation == 'top_down':
            strategy = RandomSearchStrategyTopDown(problem)
        else:
            strategy = RandomSearchStrategyBottomUp(problem)
    else:
        raise ValueError(f"Unknown synthesis strategy: {config.synthesis_parameters.strategy}")

    print(strategy.problem.info_smt())
    strategy.execute_cegis()

if __name__ == '__main__':
    main()
