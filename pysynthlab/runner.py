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

    x, y = Ints('x y')
    f = Function('f', IntSort(), IntSort(), IntSort())
    constraints = [f(x, y) == f(y, x), And(x <= f(x, y), y <= f(x, y))]

    # Create solvers
    verifier = Solver()
    enumerator = Solver()

    # Adding the original problem constraints to the verifier
    verifier.add(constraints)

    # Guess 1: f as x
    print("Guess 1: f(x, y) = x")
    enumerator.push()
    enumerator.add(f(x, y) == x)
    print("Enumerator Solver State:", enumerator.to_smt2())
    if enumerator.check() == sat:
        print("Candidate satisfies counterexamples")
        verifier.push()
        verifier.add(f(x, y) == x)
        print("Verifier Solver State:", verifier.to_smt2())
        if verifier.check() == sat:
            m = verifier.model()
            print("Counterexample found:")
            print(m)
            enumerator.pop()
            enumerator.add(f(m[x], m[y]) != m.evaluate(f(m[x], m[y])))
            print("Enumerator Solver State:", enumerator.to_smt2())
        verifier.pop()
    else:
        print("Candidate does not satisfy counterexamples")

    # Guess 2: f as constant 0
    print("Guess 2: f(x, y) = 0")
    enumerator.push()
    enumerator.add(f(x, y) == 0)
    print("Enumerator Solver State:", enumerator.to_smt2())
    if enumerator.check() == sat:
        print("Candidate satisfies counterexamples")
        verifier.push()
        verifier.add(f(x, y) == 0)
        print("Verifier Solver State:", verifier.to_smt2())
        if verifier.check() == sat:
            m = verifier.model()
            print("Counterexample found:")
            print(m)
            enumerator.pop()
            enumerator.add(f(m[x], m[y]) != m.evaluate(f(m[x], m[y])))
            print("Enumerator Solver State:", enumerator.to_smt2())
        verifier.pop()
    else:
        print("Candidate does not satisfy counterexamples")

    # Guess 3: correct solution
    print("Guess 3: f(x, y) = If(x <= y, y, x)")
    enumerator.push()
    enumerator.add(f(x, y) == If(x <= y, y, x))
    print("Enumerator Solver State:", enumerator.to_smt2())
    if enumerator.check() == sat:
        print("Candidate satisfies counterexamples")
        verifier.push()
        verifier.add(f(x, y) == If(x <= y, y, x))
        print("Verifier Solver State:", verifier.to_smt2())
        if verifier.check() == sat:
            m = verifier.model()
            print("Counterexample found:")
            print(m)
            enumerator.pop()
            enumerator.add(f(m[x], m[y]) != m.evaluate(f(m[x], m[y])))
            print("Enumerator Solver State:", enumerator.to_smt2())
        else:
            print("Valid solution found with the third guess")
        verifier.pop()
    else:
        print("Candidate does not satisfy counterexamples")

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
