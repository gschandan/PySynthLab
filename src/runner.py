import argparse
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, FileType

from src.cegis.z3.fast_enumerative_synth_bottom_up import FastEnumerativeSynthesisBottomUp
from src.cegis.z3.fast_enumerative_synth_top_down import FastEnumerativeSynthesisTopDown
from src.cegis.z3.random_search_bottom_up import SynthesisProblem, RandomSearchStrategyBottomUp
from src.cegis.z3.random_search_bottom_up_cegis_t import RandomSearchStrategyBottomUpCegisT
from src.cegis.z3.random_search_top_down import RandomSearchStrategyTopDown
from src.cegis.z3.synthesis_problem import SynthesisProblemOptions


def main(args: argparse.Namespace) -> None:
    """
    Main function to run the synthesis problem.

    :param args: Command-line arguments. 
    """

    file_content = args.input_file.read()
    random_seed = SynthesisProblemOptions.random_seed

    if args.random_seed_behaviour == 'fixed':
        random_seed = args.random_seed
    elif args.random_seed_behaviour == 'random':
        random_seed = None

    options = SynthesisProblemOptions(
        sygus_standard=int(args.sygus_standard),
        verbose=args.verbose,
        min_const=args.min_const,
        max_const=args.max_const,
        max_depth=args.max_depth,
        max_complexity=args.max_complexity,
        random_seed=random_seed,
        randomise_each_iteration=args.randomise_each_iteration,
        max_candidates_at_each_depth=args.max_candidates_at_each_depth
    )
    problem = SynthesisProblem(file_content, options)

    if args.strategy == 'fast_enumerative_bottom_up':
        strategy = FastEnumerativeSynthesisBottomUp(problem)
    elif args.strategy == 'fast_enumerative_top_down':
        strategy = FastEnumerativeSynthesisTopDown(problem)
    elif args.strategy == 'random_search_bottom_up':
        strategy = RandomSearchStrategyBottomUp(problem)
    elif args.strategy == 'random_search_top_down':
        strategy = RandomSearchStrategyTopDown(problem) 
    elif args.strategy == 'cegis_t_bottom_up':
        strategy = RandomSearchStrategyBottomUpCegisT(problem)
    else:
        raise ValueError(f"Unknown synthesis strategy: {args.strategy}")

    print(strategy.problem.info_smt())
    strategy.execute_cegis()


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '-v', '--verbose', type=int, default=0, choices=[0, 1, 2],
        help='Debugging message suppression:\n'
             '\t 0 = no suppression; all output printed to console\n'
             '\t 1 = suppress warnings\n'
             '\t 2 = suppress all output except success/failure')

    parser.add_argument(
        '-s', '--sygus-standard', default='2', choices=['1', '2'],
        help='The SyGuS language standard used in the input file')

    parser.add_argument(
        'input_file', type=FileType('r'),
        help='Path to an input file (or stdin if "-")')

    parser.add_argument(
        '--strategy', type=str, default='fast_enumerative_bottom_up',
        choices=['fast_enumerative_bottom_up', 'fast_enumerative_top_down', 'random_search_bottom_up',
                 'random_search_top_down', 'cegis_t_bottom_up'],
        help='The synthesis strategy to use')

    parser.add_argument(
        '--min-const', type=int, default=SynthesisProblemOptions.min_const,
        help='Minimum constant value to include in expressions for candidate synthesis')

    parser.add_argument(
        '--max-const', type=int, default=SynthesisProblemOptions.max_const,
        help='Maximum constant value to include in expressions for candidate synthesis')

    parser.add_argument(
        '--max-depth', type=int, default=SynthesisProblemOptions.max_depth,
        help='Maximum depth for candidate generation')

    parser.add_argument(
        '--max-candidates-at-each-depth', type=int, default=SynthesisProblemOptions.max_candidates_at_each_depth,
        help='Maximum number of candidates to evaluate at each depth for random search strategy')

    parser.add_argument(
        '--max-complexity', type=int, default=SynthesisProblemOptions.max_complexity,
        help='Maximum complexity for the random search strategy')

    parser.add_argument(
        '--random-seed-behaviour', type=str, default='fixed', choices=['fixed', 'random'],
        help='Behaviour for the random seed in the random search strategy:\n'
             '\t fixed = use the provided random seed on every iteration\n'
             '\t random = generate a new random seed for each iteration')

    parser.add_argument(
        '--random-seed', type=int, default=SynthesisProblemOptions.random_seed,
        help='Random seed for the random search strategy (used when random-seed-behavior is set to "fixed")')

    parser.add_argument(
        '--randomise-each-iteration', action='store_true',
        help='Randomise the random seed for each iteration in the random search strategy')

    main(parser.parse_args())
