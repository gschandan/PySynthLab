import itertools

import z3

from pysynthlab.synthesis_problem import SynthesisProblem
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, FileType


def main(args):
    file = args.input_file.read()

    solver = z3.Solver()
    #solver.set("timeout", 30)
    problem = SynthesisProblem(file, solver, int(args.sygus_standard))
    problem.info()
    print(problem.get_logic())

    solver.add(z3.parse_smt2_string(problem.convert_sygus_to_smt()))
    depth = 0
    depth_limit = 10  # prevent excessive search depth
    breadth_limit = 20  # limit the number of expressions evaluated at each depth
    size_limit = 5
    counterexample_history = set()

    func_name, z3_func = list(problem.z3functions.items())[0]
    func = problem.get_synth_func(func_name)
    arg_sorts = [problem.convert_sort_descriptor_to_z3_sort(sort_descriptor) for sort_descriptor in func.argument_sorts]
    func_args = {name: z3.Const(name, sort) for name, sort in zip(func.argument_names, arg_sorts)}
    func_arg_values = func_args.values()

    while True:
        candidate_expressions = problem.generate_linear_integer_expressions_v2(depth)
        for candidate_expr in itertools.islice(candidate_expressions, breadth_limit):
            solver.push()
            solver.add(z3.parse_smt2_string(problem.extract_synth_function(func_name)))

            solver.add(z3_func(*func_arg_values) == candidate_expr)
            print("Candidate Expr:", candidate_expr)
            result = solver.check()
            if result == z3.sat:
                model = solver.model()
                print("model", model)
                counterexample = {var: model.evaluate(var) for var in func_arg_values}

                # de-dupe counterexamples
                counterexample_key = tuple(counterexample.items())
                if counterexample_key not in counterexample_history:
                    print('Counterexample:', counterexample)
                    counterexample_history.add(counterexample_key)
                    additional_constraints = [var != counterexample[var] for var in func_arg_values]
                    print("Additional Constraints:", additional_constraints)
                    solver.add(*additional_constraints)
                solver.pop()
            else:
                print('Valid candidate found:', candidate_expr)
                solver.pop()
                return candidate_expr
        depth += 1
        print("Depth: ", depth)
        if depth > depth_limit:
            print("Depth limit reached without finding a valid candidate.")
            break


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
