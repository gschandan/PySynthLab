from z3 import *

from pysynthlab.synthesis_problem import SynthesisProblem
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, FileType


def main(args):
    file = args.input_file.read()

    problem = SynthesisProblem(file, int(args.sygus_standard))
    parsed_sygus_problem = problem.convert_sygus_to_smt()
    problem.counterexample_solver.add(z3.parse_smt2_string(parsed_sygus_problem))
    problem.verification_solver.add(z3.parse_smt2_string(parsed_sygus_problem))
    problem.info()
    print(parsed_sygus_problem)

    depth = 0
    itr = 0
    depth_limit = 200
    found_valid_candidate = False

    assertions = problem.counterexample_solver.assertions()
    problem.assertions.update(assertions)
    for assertion in assertions:
        problem.original_assertions.append(assertion)

    problem.counterexample_solver.reset()

    negated_assertions = problem.negate_assertions(assertions)
    problem.counterexample_solver.add(*negated_assertions)
    problem.negated_assertions.update(negated_assertions)

    problem.counterexample_solver.set("timeout", 30000)
    problem.verification_solver.set("timeout", 30000)
    set_param("smt.random_seed", 1234)
    candidate_expressions = problem.generate_linear_integer_expressions(depth)
    expression = None

    while not found_valid_candidate:
        try:
            candidate_expression = next(candidate_expressions)
        except StopIteration:
            depth += 1
            if depth > depth_limit:
                print("Depth limit reached without finding a valid candidate.")
                break
            candidate_expressions = problem.generate_linear_integer_expressions(depth)
            candidate_expression = next(candidate_expressions)

        expression = problem.z3_func(*problem.func_args) == candidate_expression # (ite (<= x y) y x)
        if itr > 100:
            p = list(problem.z3variables.values())
            expression = problem.z3_func(*problem.func_args) == z3.If(p[0] <= p[1], p[1], p[0])
        if expression in problem.assertions:
            itr += 1
            continue
        print("expr:", expression)

        # counterexample generation
        problem.counterexample_solver.push()
        problem.counterexample_solver.add(expression)
        counterexample_result = problem.counterexample_solver.check()
        print("Counterexample result:", counterexample_result)
        if counterexample_result == z3.sat:
            model = problem.counterexample_solver.model()
            print("Counterexample model:", model)
            counterexample = problem.check_counterexample(model)
            if counterexample is not None:
                problem.counterexample_solver.pop()
                additional_constraints = problem.get_additional_constraints(counterexample)
                problem.additional_constraints.append(additional_constraints)
                problem.counterexample_solver.add(additional_constraints)
                problem.assertions.add(expression)
            else:
                problem.counterexample_solver.pop()
        else:
            problem.counterexample_solver.pop()

        # verification
        if counterexample_result == z3.unsat:
            problem.verification_solver.push()
            problem.verification_solver.add(expression)
            verifier_result = problem.verification_solver.check()
            print("Verification result:", verifier_result)
            if verifier_result == z3.sat:
                found_valid_candidate = True
            problem.verification_solver.pop()

        itr += 1
        print(f"Depth {depth}, Iteration {itr}")

    if found_valid_candidate:
        print("Expression:", expression)
    else:
        print("No valid candidate found within the depth/loop/time limit.")

    print("COUNTEREXAMPLE SMT: ", problem.counterexample_solver.to_smt2())
    print("CE Stats: ", problem.counterexample_solver.statistics())
    print("VERIFICATION SMT: ", problem.verification_solver.to_smt2())
    print("VERIFIER Stats: ", problem.verification_solver.statistics())


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
