import argparse
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, FileType
from pysynthlab.cegis.z3.synthesis_problem_z3 import SynthesisProblemOptions, SynthesisProblem


def main(args: argparse.Namespace) -> None:
    """
    Main function to run the synthesis problem.

    :param args: Command-line arguments.
    """

    file_content = args.input_file.read()

    options = SynthesisProblemOptions(
        sygus_standard=int(args.sygus_standard),
        verbose=args.verbose
    )

    problem = SynthesisProblem(file_content, options)

    if options.verbose < 2:
        problem.info()

    problem.execute_cegis()


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '-v', '--verbose', type=int, default=0, choices=[0, 1, 2],
        help='Verbosity level:\n'
             '\t 0 = no suppression; all output printed to console\n'
             '\t 1 = suppress warnings\n'
             '\t 2 = suppress all output except success/failure')

    parser.add_argument(
        '-s', '--sygus-standard', default='2', choices=['1', '2'],
        help='The SyGuS language standard used in the input file')

    parser.add_argument(
        'input_file', type=FileType('r'),
        help='Path to an input file (or stdin if "-")')

    main(parser.parse_args())
