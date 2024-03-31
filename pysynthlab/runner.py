from z3 import *
import pyparsing

from pysynthlab.synthesis_problem import SynthesisProblem
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, FileType


def manual_loops():
    def add_negated_constraints(solver, f_guess):
        x, y = Ints('x y')
        f_x_y = f_guess(x, y)
        f_y_x = f_guess(y, x)
        solver.add(Or(Not(f_x_y == f_y_x), Not(And(x <= f_x_y, y <= f_x_y))))

    def add_original_constraints(solver, f_guess):
        x, y = Ints('x y')
        f_x_y = f_guess(x, y)
        f_y_x = f_guess(y, x)
        solver.add(And((f_x_y == f_y_x), And(x <= f_x_y, y <= f_x_y)))

    guesses = [
        (lambda a, b: 0, "f(x, y) = 0"),
        (lambda a, b: a, "f(x, y) = x"),
        (lambda a, b: b, "f(x, y) = y"),
        (lambda a, b: a - b, "f(x, y) = x - y"),
        (lambda a, b: If(a <= b, b, a), "f(x, y) = max(x, y)"),
        (lambda a, b: If(a <= b, a, b), "f(x, y) = min(x, y)"),

    ]

    for guess, name in guesses:
        enumerator = Solver()
        add_negated_constraints(enumerator, guess)
        print("ENUMAERATOR:", enumerator.to_smt2())

        if enumerator.check() == sat:
            model = enumerator.model()
            print(
                f"Counterexample for guess {name}: x = {model.evaluate(Int('x'))}, y = {model.evaluate(Int('y'))}")

            verifier = Solver()
            add_original_constraints(verifier, guess)
            verifier.add(Int('x') == model[Int('x')], Int('y') == model[Int('y')])
            print("VERIFIER:", verifier.to_smt2())

            if verifier.check() == sat:
                print(f"Verification passed unexpectedly for guess {name}. Possible error in logic.")
            else:
                print(f"Verification failed for guess {name}, counterexample confirmed.")
        else:
            verifier = Solver()
            add_original_constraints(verifier, guess)
            print("VERIFIER:", verifier.to_smt2())
            if verifier.check() == sat:
                print(f"No counterexample found for guess {name}. Guess should be correct.")
            else:
                print(f"Verification failed unexpectedly for guess {name}. Possible error in logic.")
        print("-" * 50)

def main(args):

    manual_loops()
    file = args.input_file.read()

    problem = SynthesisProblem(file, int(args.sygus_standard))
    parsed_sygus_problem = problem.convert_sygus_to_smt()
    problem.info()
    print(parsed_sygus_problem)




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
