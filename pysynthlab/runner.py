import itertools
from functools import lru_cache

import z3

from pysynthlab.fast_enumerative_synthesis import FastEnumSynth
from pysynthlab.synthesis_problem import SynthesisProblem
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, FileType


def main(args):
    file = args.input_file.read()

    solver = z3.Solver()
    solver.set("timeout", 30000)
    problem = SynthesisProblem(file, solver, int(args.sygus_standard))
    problem.info()
    print(problem.get_logic())

    solver.add(z3.parse_smt2_string(problem.convert_sygus_to_smt()))
    depth = 0
    depth_limit = 50  # prevent excessive search depth
    breadth_limit = 50  # limit the number of expressions evaluated at each depth
    size_limit = 15

    func_name, z3_func = list(problem.z3functions.items())[0]
    func = problem.get_synth_func(func_name)
    arg_sorts = [problem.convert_sort_descriptor_to_z3_sort(sort_descriptor) for sort_descriptor in func.argument_sorts]
    func_args = [z3.Const(name, sort) for name, sort in zip(func.argument_names, arg_sorts)]
    found_valid_candidate = False

    while not found_valid_candidate:
        candidate_expressions = problem.generate_linear_integer_expressions(depth, size_limit)

        for candidate_expr in itertools.islice(candidate_expressions, breadth_limit):
            solver.push()
            expr = z3_func(*func_args) == candidate_expr
            solver.add(expr)
            print("Expr:", expr)
            result = solver.check()

            if result == z3.sat:
                model = solver.model()
                print("model", model)
                counterexample = {str(var): model.evaluate(var, model_completion=True) for var in func_args}
                print("Counterexample:", counterexample)
                additional_constraints = [var != counterexample[str(var)] for var in func_args]
                solver.add(*additional_constraints)
            else:
                validation_solver = z3.Solver()
                validation_solver.add(solver.assertions())
                validation_result = validation_solver.check()
                if validation_result == z3.sat:
                    found_valid_candidate = True
                    best_candidate = candidate_expr
                    break

            solver.pop()

        if found_valid_candidate:
            break

        depth += 1
        print("Depth: ", depth)
        if depth > depth_limit:
            print("Depth limit reached without finding a valid candidate.")
            break
    if found_valid_candidate:
        print("Best candidate:", best_candidate)
        return best_candidate
    else:
        print("No valid candidate found within the depth limit.")
        return None

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
