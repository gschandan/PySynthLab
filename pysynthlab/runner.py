from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, FileType
import z3

from pysynthlab.helpers.parser.src.ast import CommandKind, ASTVisitor
from pysynthlab.synthesis_problem import SynthesisProblem


def main(args):
    file = args.input_file.read()

    solver = z3.Solver()
    solver.set("timeout", 5)
    problem = SynthesisProblem(file, solver, int(args.sygus_standard))
    problem.info()
    print(problem.get_logic())
    problem.setup_solver()

    # if solver.check() == z3.sat:
    #     model = solver.model()
    #     print(model)
    #
    #     for constraint in solver.assertions():
    #         solver.add(z3.Not(constraint))
    #         solver.push()
    #         print(solver.assertions())
    #         result = solver.check()
    #         if result == z3.sat:
    #             print("Not a valid invariant. Counter-example:")
    #             print(solver.model())
    #         elif result == z3.unsat:
    #             print("Invariant is valid")
    #         else:
    #             print(result)
    #
    #     # negated_constraints = []
    #     # for var in model:
    #     #     variable_name = str(var)
    #     #     variable_value = model[var]
    #     #     if z3.is_int_value(variable_value):
    #     #         negated_constraints.append(z3.Or(z3.Int(variable_name) != variable_value))
    #     # solver.add(negated_constraints)
    #
    #     print(model)
    #     print(solver.statistics())


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
