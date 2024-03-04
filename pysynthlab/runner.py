import itertools
import time
from functools import lru_cache

from z3 import *

from pysynthlab.fast_enumerative_synthesis import FastEnumSynth
from pysynthlab.synthesis_problem import SynthesisProblem
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, FileType


def main(args):
    file = args.input_file.read()

    problem = SynthesisProblem(file, int(args.sygus_standard))
    solver = problem.solver
    parsed_sygus_problem = problem.convert_sygus_to_smt()
    solver.add(z3.parse_smt2_string(parsed_sygus_problem))
    problem.info()
    print(parsed_sygus_problem)

    depth = 0
    itr = 0;
    depth_limit = 200
    found_valid_candidate = False
    candidate_expression = None
    assertions = problem.solver.assertions()
    solver.reset()

    solver.add(problem.negate_assertions(assertions))

    solver.set("timeout", 30000)
    set_param("smt.random_seed", 1234)
    candidate_expressions = problem.generate_linear_integer_expressions(depth)

    while not found_valid_candidate:
        #print("CURRENT_ASSERTIONS: \n", problem.solver.assertions());
        try:
            candidate_expression = next(candidate_expressions)
        except StopIteration:
            depth += 1
            if depth > depth_limit:
                print("Depth limit reached without finding a valid candidate.")
                break
            candidate_expressions = problem.generate_linear_integer_expressions(depth)
            candidate_expression = next(candidate_expressions)

        solver.push()
        #p = list(problem.z3variables.values())
        #expression = problem.z3_func(*problem.func_args) == z3.If(p[0] <= p[1], p[1], p[0]) # (ite (<= x y) y x) - correct
        expression = problem.z3_func(*problem.func_args) == candidate_expression # (ite (<= x y) y x)
        print("expr:", expression)
        print("SMT: ", solver.to_smt2())
        solver.add(expression)
        result = solver.check()
        print(result)
        if result == z3.sat:
            model = solver.model()
            print("Candidate model:", model)

            counterexample = problem.check_counterexample(model)
            if counterexample is None:
                found_valid_candidate = True
                break
            else:
                solver.pop()
                additional_constraints = [problem.z3_func(*problem.func_args) != candidate_expression, *problem.get_additional_constraints(counterexample)]
                print(additional_constraints)
                solver.add(*additional_constraints)
        else:
            found_valid_candidate = True
            break

        itr += 1
        print(f"Depth {depth}, Iteration {itr}")

    if found_valid_candidate:
        print("Expression:", expression)
    else:
        print("No valid candidate found within the depth/loop/time limit.")

    print("SMT: ", solver.to_smt2())
    print("Stats: ", solver.statistics())

    # fast_enum_synth = FastEnumSynth(problem)
    # fast_enum_synth.generate(max_depth=10)


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
