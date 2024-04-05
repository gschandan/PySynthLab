from z3 import *
import cvc5
import pyparsing

from pysynthlab.synthesis_problem import SynthesisProblem
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

def main(args):
    manual_loops()
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
        not_monotonic = solver.mkTerm(cvc5.Kind.NOT, solver.mkTerm(cvc5.Kind.AND, solver.mkTerm(cvc5.Kind.LEQ, x, f_x_y), solver.mkTerm(cvc5.Kind.LEQ, y, f_x_y)))
        solver.assertFormula(solver.mkTerm(cvc5.Kind.OR, not_equal, not_monotonic))

    def add_original_constraints(solver, x, y, f_guess):
        f_x_y = f_guess(x, y)
        f_y_x = f_guess(y, x)
        equal = solver.mkTerm(cvc5.Kind.EQUAL, f_x_y, f_y_x)
        monotonic = solver.mkTerm(cvc5.Kind.AND, solver.mkTerm(cvc5.Kind.LEQ, x, f_x_y), solver.mkTerm(cvc5.Kind.LEQ, y, f_x_y))
        solver.assertFormula(solver.mkTerm(cvc5.Kind.AND, equal, monotonic))

    def lambda_to_cvc5_term(solver, lambda_expr, a, b):
        if lambda_expr == 0:
            return solver.mkInteger(0)
        elif lambda_expr == a:
            return a
        elif lambda_expr == b:
            return b
        elif "max" in str(lambda_expr):
            return solver.mkTerm(cvc5.Kind.ITE, solver.mkTerm(cvc5.Kind.LEQ, a, b), b, a)
        else:
            if lambda_expr == a - b:
                return solver.mkTerm(cvc5.Kind.MINUS, a, b)
            raise ValueError(f"Unsupported lambda expression: {lambda_expr}")

    guesses = [
        (lambda a, b: solver.mkInteger(0), "f(x, y) = 0"),
        (lambda a, b: a, "f(x, y) = x"),
        (lambda a, b: b, "f(x, y) = y"),
        (lambda a, b: solver.mkTerm(cvc5.Kind.SUB, a, b), "f(x, y) = x - y"),
        (lambda a, b: solver.mkTerm(cvc5.Kind.ITE, solver.mkTerm(cvc5.Kind.LEQ, a, b), b, a), "f(x, y) = max(x, y)"),
    ]

    for guess_func, name in guesses:
        solver, x, y = create_solver_with_vars()
        add_negated_constraints(solver, x, y, guess_func)

        result = solver.checkSat()
        if result.isSat():
            model = solver.getModel([], [x,y])
            print(f"Model for guess {name}: x = {model[0]}, y = {model[1]}")
        else:
            print(f"No model for guess {name}.")

        print("-" * 50)

# def main(args):
#     #manual_loops()
#     file = args.input_file.read()
#
#     problem = SynthesisProblem(file, int(args.sygus_standard))
#     parsed_sygus_problem = problem.convert_sygus_to_smt()
#     problem.info()
#     print(parsed_sygus_problem)
#
#     depth = 0
#     itr = 0
#     depth_limit = 200
#     found_valid_candidate = False
#
#     problem.verification_solver.add(z3.parse_smt2_string(parsed_sygus_problem))
#
#     assertions = problem.verification_solver.assertions()
#     problem.assertions.update(assertions)
#     for assertion in assertions:
#         problem.original_assertions.append(assertion)
#
#     problem.enumerator_solver.reset()
#
#     negated_assertions = problem.negate_assertions(assertions)
#     problem.enumerator_solver.add(*negated_assertions)
#     problem.negated_assertions.update(negated_assertions)
#
#     # problem.counterexample_solver.set("timeout", 30000)
#     # problem.verification_solver.set("timeout", 30000)
#
#     set_param("smt.random_seed", 1234)
#     candidate_functions = problem.generate_candidate_functions(depth)
#     candidate_function = None
#
#     while not found_valid_candidate and itr < 101:
#         try:
#             candidate_function = next(candidate_functions)
#         except StopIteration:
#             depth += 1
#             if depth > depth_limit:
#                 print("Depth limit reached without finding a valid candidate.")
#                 break
#             candidate_functions = problem.generate_candidate_functions(depth)
#             candidate_function = next(candidate_functions)
#
#         if itr == 100:
#             p = list(problem.z3variables.values())
#             candidate_function = lambda a, b: If(a <= b, b, a)
#         if candidate_function in problem.assertions:
#             itr += 1
#             continue
#         print("func:", candidate_function)
    #
    #     problem.enumerator_solver.push()
    #     problem.enumerator_solver.add(expression)
    #     enumerator_solver_result = problem.enumerator_solver.check()
    #     print("Verification result:", enumerator_solver_result)
    #     problem.enumerator_solver.pop()
    #     model = problem.enumerator_solver.model()
    #     counterexample = problem.check_counterexample(model)
    #     if counterexample is not None:
    #         additional_constraint = problem.get_additional_constraints(counterexample)
    #         problem.enumerator_solver.add(additional_constraint)
    #     itr += 1
    #     print(f"Depth {depth}, Iteration {itr}")
    #
    # if found_valid_candidate:
    #     print("VALID CANDIDATE:", expression)
    # else:
    #     print("No valid candidate found within the depth/loop/time limit.")
    #
    # print("VERIFICATION SMT: ", problem.verification_solver.to_smt2())
    # print("COUNTEREXAMPLE SMT: ", problem.enumerator_solver.to_smt2())

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
