import itertools

import z3

from pysynthlab.synthesis_problem import SynthesisProblem
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, FileType


def main(args):
    file = args.input_file.read()

    solver = z3.Solver()
    solver.set("timeout", 30)
    problem = SynthesisProblem(file, solver, int(args.sygus_standard))
    problem.info()
    print(problem.get_logic())

    solver.add(z3.parse_smt2_string(problem.convert_sygus_to_smt()))
    depth = 0
    depth_limit = 50 # prevent excessive search depth
    breadth_limit = 100  # limit the number of expressions evaluated at each depth
    size_limit = 5
    counterexample_history = set()

    while True:
        candidate_expressions = problem.generate_linear_integer_expressions_v4(depth, size_limit)

        for candidate_expr in itertools.islice(candidate_expressions, breadth_limit):
            for func_name, z3_func in problem.z3functions.items():
                func_args = [z3.Int(f'arg_{i}') for i in range(z3_func.arity()) if z3_func.domain(i) == z3.IntSort()]

            solver.push()
            solver.add(z3.parse_smt2_string(problem.extract_synth_function(func_name)))
            solver.add(z3_func(*func_args) == candidate_expr)

            result = solver.check()
            if result == z3.sat:
                model = solver.model()
                counterexample = tuple(
                    sorted((str(var), model[var]) for var in model.decls() if str(var) in problem.z3variables))

                # de-dupe counterexamples
                if counterexample not in counterexample_history:
                    print('Counterexample:', dict(counterexample))
                    counterexample_history.add(counterexample)
                    additional_constraints = [z3_var != model[z3_var] for z3_var in func_args if z3_var in model]
                    solver.add(*additional_constraints)
                solver.pop()
            else:
                print('Valid candidate found:', candidate_expr)
                solver.pop()
                return candidate_expr
            print(candidate_expr)
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
