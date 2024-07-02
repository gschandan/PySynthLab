from dataclasses import asdict
import argparse
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, FileType
from src.cegis.z3.config_manager import ConfigManager
from src.cegis.z3.fast_enumerative_synth_bottom_up import FastEnumerativeSynthesisBottomUp
from src.cegis.z3.fast_enumerative_synth_top_down import FastEnumerativeSynthesisTopDown
from src.cegis.z3.random_search_bottom_up import RandomSearchStrategyBottomUp
from src.cegis.z3.random_search_bottom_up_cegis_t import RandomSearchStrategyBottomUpCegisT
from src.cegis.z3.random_search_top_down import RandomSearchStrategyTopDown
from src.cegis.z3.synthesis_problem import SynthesisProblem


def main() -> None:
    config = ConfigManager.get_config()
    print(asdict(config))

    problem = """
    (set-logic LIA)
    (synth-fun max2 ((a Int) (b Int)) Int)

    (declare-var x Int)
    (declare-var y Int)

    (constraint (>= (max2 x y) x))
    (constraint (>= (max2 x y) y))
    (constraint (or (= x (max2 x y)) (= y (max2 x y))))
    (constraint (= (max2 x x) x))

    (constraint (forall ((x Int) (y Int))
      (=> (>= x y) (= (max2 x y) x))))
    (constraint (forall ((x Int) (y Int))
      (=> (>= y x) (= (max2 y x) y))))
    (check-synth)
    """

    problem = SynthesisProblem(problem, config)

    if config.synthesis_parameters_strategy == 'fast_enumerative_bottom_up':
        strategy = FastEnumerativeSynthesisBottomUp(problem)
    elif config.synthesis_parameters_strategy == 'fast_enumerative_top_down':
        strategy = FastEnumerativeSynthesisTopDown(problem)
    elif config.synthesis_parameters_strategy == 'random_search_bottom_up':
        strategy = RandomSearchStrategyBottomUp(problem)
    elif config.synthesis_parameters_strategy == 'random_search_top_down':
        strategy = RandomSearchStrategyTopDown(problem)
    elif config.synthesis_parameters_strategy == 'cegis_t_bottom_up':
        strategy = RandomSearchStrategyBottomUpCegisT(problem)
    else:
        strategy = RandomSearchStrategyBottomUp(problem)
    # else:
    #     raise ValueError(f"Unknown synthesis strategy: {config.synthesis_parameters_strategy}")

    print(strategy.problem.info_smt())
    strategy.execute_cegis()


if __name__ == '__main__':
    main()
