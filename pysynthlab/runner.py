import warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, FileType

from pysynthlab.synthesis_problem import SynthesisProblem


def main(args):
    file = args.input_file.read()
    problem = SynthesisProblem(file, int(args.sygus_standard))
    problem.info()
    print(problem.get_logic())


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '-b', '--binarize', action='store_true',
        help='Convert all chainable operators to binary operator applications')
    parser.add_argument(
        '-q', '--quiet', action='store_true',
        help='Suppress all messages and debugging output')
    parser.add_argument(
        '-u', '--no-unary-minus', action='store_true',
        help='Convert all (- x) terms to (- 0 x)')

    parser.add_argument(
        '-s', '--sygus-standard', default='2', choices=['1','2'],
        help='The SyGuS language standard used in the input file')

    parser.add_argument(
        'input_file', type=FileType('r'),
        help='Path to an input file (or stdin if "-")')

    main(parser.parse_args())
