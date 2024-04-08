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

    file = args.input_file.read()

    problem = SynthesisProblemCvc5(file, int(args.sygus_standard))
    parsed_sygus_problem = problem.convert_sygus_to_smt()
    problem.info()
    print(parsed_sygus_problem)

    depth = 0
    itr = 0
    depth_limit = 200
    found_valid_candidate = False

    # setup solvers
    verification_solver = problem.verification_solver
    # add variables
    for variable_name, variable in problem.cvc5variables.items():
        verification_solver.mkConst(variable.getSort(),variable_name)
    # add constraints
    for constraint in problem.cvc5_constraints:
        verification_solver.addSygusConstraint(constraint)

    constraints = problem.verification_solver.getSygusConstraints()
    problem.assertions.update(constraints)
    for constraint in constraints:
        problem.original_assertions.append(constraint)
    print("Constraints", constraints)
    enumerator_solver = problem.enumerator_solver
    # add variables
    for variable_name, variable in problem.cvc5variables.items():
        enumerator_solver.mkConst(variable.getSort(), variable_name)
    # negate and add to enumerator solver
    negated_assertions = problem.negate_assertions(constraints, enumerator_solver)
    for constraint in negated_assertions:
        problem.enumerator_solver.assertFormula(constraint)
    problem.negated_assertions.update(negated_assertions)

    candidate_functions = problem.generate_candidate_functions(depth)
    candidate_function = None

    while not found_valid_candidate and itr < 101:
        try:
            candidate_function = next(candidate_functions)
        except StopIteration:
            depth += 1
            if depth > depth_limit:
                print("Depth limit reached without finding a valid candidate.")
                break
            candidate_functions = problem.generate_candidate_functions(depth)
            candidate_function = next(candidate_functions)

        if itr == 100:
            p = list(problem.cvc5variables.values())
            def candidate_function(a, b): problem.enumerator_solver.mkTerm(Kind.ITE,problem.enumerator_solver.mkTerm(Kind.LEQ, a, b),b, a)

        if candidate_function in problem.assertions:
            itr += 1
            continue
        print("func:", candidate_function)

        problem.enumerator_solver.push()
        problem.enumerator_solver.assertFormula(candidate_function(*problem.func_args))
        enumerator_solver_result = problem.enumerator_solver.checkSat()
        print("Verification result:", enumerator_solver_result)
        problem.enumerator_solver.pop()

        if enumerator_solver_result.isSat():
            model = problem.enumerator_solver.getModel([], problem.func_args)
            counterexample = problem.check_counterexample(model)
            if counterexample is not None:
                additional_constraint = problem.get_additional_constraints(counterexample)
                problem.enumerator_solver.assertFormula(additional_constraint)
            else:
                found_valid_candidate = True
        itr += 1
        print(f"Depth {depth}, Iteration {itr}")

    if found_valid_candidate:
        print("VALID CANDIDATE:", candidate_function)
    else:
        print("No valid candidate found within the depth/loop/time limit.")

    print("VERIFICATION SMT: ", problem.verification_solver.toString())
    print("COUNTEREXAMPLE SMT: ", problem.enumerator_solver.toString())



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
