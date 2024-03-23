from z3 import *

from pysynthlab.synthesis_problem import SynthesisProblem
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, FileType


def main(args):
    # file = args.input_file.read()
    #
    # problem = SynthesisProblem(file, int(args.sygus_standard))
    # parsed_sygus_problem = problem.convert_sygus_to_smt()
    # problem.info()
    # print(parsed_sygus_problem)

    base_problem = """
    (declare-fun x () Int)
    (declare-fun y () Int)
    (assert(or (not(= (f x y) (f y x))) (not (and (<= x (f x y)) (<= y (f x y))))))
    """

    guesses = [
        "(define-fun f ((x Int) (y Int)) Int x)",  # Guess 1: f(x, y) = x
        "(define-fun f ((x Int) (y Int)) Int y)",  # Guess 2: f(x, y) = y
        "(define-fun f ((x Int) (y Int)) Int (ite (<= x y) y x))"  # Guess 3: f(x, y) = max(x, y)
    ]

    def try_guess(base_problem, guess):
        smt_lib_str = guess + base_problem + "(check-sat)(get-model)"
        solver = z3.Solver()
        solver.from_string(smt_lib_str)
        print("SMT:", solver.to_smt2())
        if solver.check() == z3.sat:
            print(f"Guess '{guess}' is not valid, found counterexample:")
            print(solver.model())
            return False
        else:
            print(f"Guess '{guess}' is potentially correct, no counterexample found.")
            return True

    for guess in guesses:
        if try_guess(base_problem, guess):
            break

    # depth = 0
    # itr = 0
    # depth_limit = 200
    # found_valid_candidate = False
    #
    # problem.verification_solver.add(z3.parse_smt2_string(parsed_sygus_problem))
    #
    # assertions = problem.verification_solver.assertions()
    # problem.assertions.update(assertions)
    # for assertion in assertions:
    #     problem.original_assertions.append(assertion)
    #
    # problem.enumerator_solver.reset()
    #
    # negated_assertions = problem.negate_assertions(assertions)
    # problem.enumerator_solver.add(*negated_assertions)
    # problem.negated_assertions.update(negated_assertions)
    #
    # #problem.counterexample_solver.set("timeout", 30000)
    # #problem.verification_solver.set("timeout", 30000)
    #
    # set_param("smt.random_seed", 1234)
    # candidate_expressions = problem.generate_linear_integer_expressions(depth)
    # expression = None
    #
    # while not found_valid_candidate:
    #     try:
    #         candidate_expression = next(candidate_expressions)
    #     except StopIteration:
    #         depth += 1
    #         if depth > depth_limit:
    #             print("Depth limit reached without finding a valid candidate.")
    #             break
    #         candidate_expressions = problem.generate_linear_integer_expressions(depth)
    #         candidate_expression = next(candidate_expressions)
    #
    #     expression = problem.z3_func(*problem.func_args) == candidate_expression # (ite (<= x y) y x)
    #     if itr == 100:
    #         p = list(problem.z3variables.values())
    #         expression = problem.z3_func(*problem.func_args) == z3.If(p[0] <= p[1], p[1], p[0])
    #     if expression in problem.assertions:
    #         itr += 1
    #         continue
    #     print("expr:", expression)
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
