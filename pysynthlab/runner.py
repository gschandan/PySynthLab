from z3 import *
import cvc5
from cvc5 import Kind
import pyparsing

from pysynthlab.synthesis_problem_cvc5 import SynthesisProblemCvc5
from pysynthlab.synthesis_problem_z3 import SynthesisProblemZ3
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, FileType
import time
from typing import Optional, Tuple, List


def manual_loops():

    print("METOD 2: python function substitution")

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

    print("CVC5")

    def create_solver_with_vars():
        solver = cvc5.Solver()
        solver.setOption("produce-models", "true")
        x = solver.mkConst(solver.getIntegerSort(), "x")
        y = solver.mkConst(solver.getIntegerSort(), "y")
        return solver, x, y

    def add_negated_constraints(solver, x, y, f_guess):
        f_x_y = f_guess(x, y)
        f_y_x = f_guess(y, x)
        not_equal = solver.mkTerm(cvc5.Kind.NOT, solver.mkTerm(cvc5.Kind.EQUAL, f_x_y, f_y_x))
        not_monotonic = solver.mkTerm(cvc5.Kind.NOT,
                                      solver.mkTerm(cvc5.Kind.AND, solver.mkTerm(cvc5.Kind.LEQ, x, f_x_y),
                                                    solver.mkTerm(cvc5.Kind.LEQ, y, f_x_y)))
        solver.assertFormula(solver.mkTerm(cvc5.Kind.OR, not_equal, not_monotonic))

    guesses = [
        (lambda a, b: solver.mkInteger(0), "f(x, y) = 0"),
        (lambda a, b: a, "f(x, y) = x"),
        (lambda a, b: b, "f(x, y) = y"),
        (lambda a, b: solver.mkTerm(cvc5.Kind.SUB, a, b), "f(x, y) = x - y"),
        (
        lambda a, b: solver.mkTerm(cvc5.Kind.ITE, solver.mkTerm(cvc5.Kind.LEQ, a, b), b, a), "f(x, y) = max(x, y)"),
    ]

    for guess_func, name in guesses:
        solver, x, y = create_solver_with_vars()
        add_negated_constraints(solver, x, y, guess_func)

        result = solver.checkSat()
        if result.isSat():
            model = solver.getModel([], [x, y])
            print(f"Model for guess {name}: x = {model[0]}, y = {model[1]}")
        else:
            print(f"No model for guess {name}.")

        print("-" * 50)

def main(args):
    manual_loops()
    guesses = [
        (lambda vars, solver: solver.mkTerm(cvc5.Kind.EQUAL, solver.mkInteger(0), vars[0]), "f(...) = 0 = x"),
        (lambda vars, solver: solver.mkTerm(cvc5.Kind.EQUAL, vars[0], vars[0]), "f(x, ...) = x = x"),
        (lambda vars, solver: solver.mkTerm(cvc5.Kind.EQUAL, vars[1], vars[1]), "f(..., y) = y = y"),
        (lambda vars, solver: solver.mkTerm(cvc5.Kind.LEQ, vars[0], vars[1]), "f(x, y) = x <= y"),
        (lambda vars, solver: solver.mkTerm(cvc5.Kind.GEQ, vars[0], vars[1]), "f(x, y) = x >= y"),
    ]

    file = args.input_file.read()

    problem = SynthesisProblemCvc5(file, int(args.sygus_standard))
    parsed_sygus_problem = problem.convert_sygus_to_smt()
    problem.info()
    print(parsed_sygus_problem)

    enumerator_solver = problem.enumerator_solver
    verifier_solver = problem.verification_solver

    variables = [problem.cvc5variables[var] for var in problem.cvc5variables]

    original_constraints = problem.cvc5_constraints
    for constraint in original_constraints:
        verifier_solver.assertFormula(constraint)
    negated_constraints = problem.negate_assertions(original_constraints, enumerator_solver)
    for negated_constraint in negated_constraints:
        enumerator_solver.assertFormula(negated_constraint)

    found_valid_candidate = False
    depth = 0
    depth_limit = 200
    while not found_valid_candidate and depth < depth_limit:
        for func, description in guesses:
            candidate_expression = func(variables, enumerator_solver)
            print(f"Candidate expression {description}: {candidate_expression}")
            enumerator_solver.push()
            enumerator_solver.assertFormula(candidate_expression)
            if enumerator_solver.checkSat().isSat():
                print("Enumerator SAT")
                model = enumerator_solver.getModel([],variables)
                counterexample = problem.check_counterexample(model)
                if counterexample:
                    print("Counterexample")
                    additional_constraints = problem.get_additional_constraints(counterexample)
                    enumerator_solver.pop()
                    enumerator_solver.assertFormula(additional_constraints)
                else:
                    found_valid_candidate = True
                    enumerator_solver.pop()
                    break
            else:
                enumerator_solver.pop()

        depth += 1

    if found_valid_candidate:
        print("Valid synthesis candidate found.")
    else:
        print("No valid candidate found within the depth limit.")

    print("Verification solver state:", verifier_solver.toString())
    print("Enumerator solver state:", enumerator_solver.toString())


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
