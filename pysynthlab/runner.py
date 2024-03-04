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
    problem.info()
    print(problem.get_logic())

    solver = problem.solver
    solver.set("timeout", 30000)
    set_param("smt.random_seed", 1234)

    depth = 0
    depth_limit = 200
    loop_limit = 500
    itr = 0;
    found_valid_candidate = False
    candidate_expression = None
    while not found_valid_candidate or itr < loop_limit:

        candidate_expression = problem.generate_candidate_expression()

        solver.push()
        expr = problem.z3_func(*problem.func_args) == candidate_expression
        print("Expr:", expr)
        solver.add(expr)
        result = solver.check()

        if result == z3.sat:
            model = solver.model()
            print("Candidate model:", model)

            counterexample = problem.check_counterexample(model)
            if counterexample is None:
                found_valid_candidate = True
                break
            else:
                additional_constraints = problem.get_additional_constraints(counterexample)
                solver.add(*additional_constraints)

        solver.pop()
        itr += 1

    if depth > depth_limit:
        print("Depth limit reached without finding a valid candidate.")
    if found_valid_candidate:
        print("Best candidate:", candidate_expression)
    else:
        print("No valid candidate found within the depth limit.")

    print("Stats: ", solver.statistics())

    #fast_enum_synth = FastEnumSynth(problem)
    #fast_enum_synth.generate(max_depth=10)


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
