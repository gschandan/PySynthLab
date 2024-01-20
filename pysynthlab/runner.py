import z3

from pysynthlab.synthesis_problem import SynthesisProblem
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, FileType


def main(args):
    file = args.input_file.read()

    solver = z3.Solver()
    solver.set("timeout", 5)
    problem = SynthesisProblem(file, solver, int(args.sygus_standard))
    problem.info()
    print(problem.get_logic())

    solver.add(z3.parse_smt2_string(problem.convert_sygus_to_smt()))
    counterexamples = []
    result = solver.check()
    while True:
        if result == z3.sat:
            print('Satisfiable!')
            model = solver.model()
            print('Model:', model)
            counterexample = {var: model[var] for var in model.decls()}
            counterexamples.append(counterexample)
            print(counterexamples)
            additional_constraints = []
            for variable in problem.z3variables:
                additional_constraints.append(variable == counterexample[variable])
        else:
            print('Unsatisfiable!')
            break


if __name__ == '__main__':

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '-q', '--quiet', action='store_true',
        help='Suppress all messages and debugging output')

    parser.add_argument(
        '-s', '--sygus-standard', default='2', choices=['1', '2'],
        help='The SyGuS language standard used in the input file')

    parser.add_argument(
        'input_file', type=FileType('r'),
        help='Path to an input file (or stdin if "-")')

    main(parser.parse_args())
