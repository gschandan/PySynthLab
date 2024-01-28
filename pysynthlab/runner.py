import z3

from pysynthlab.synthesis_problem import SynthesisProblem
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, FileType


def main(args):
    file = args.input_file.read()

    solver = z3.Solver()
    solver.set("timeout", 5)
    problem = SynthesisProblem(file, solver, int(args.sygus_standard))
    problem.info()
    print(problem.get_logic())

    solver.add(z3.parse_smt2_string(problem.convert_sygus_to_smt()))
    depth = 0

    while True:
        for candidate_expr in problem.generate_linear_integer_expressions(depth):
            for func_name, z3_func in problem.z3functions.items():
                func_args = [z3.Int(f'arg_{i}') for i in range(z3_func.arity()) if z3_func.domain(i) == z3.IntSort()]

                solver.push()
                solver.add(z3.parse_smt2_string(problem.extract_synth_function(func_name)))
                solver.add(z3_func(*func_args) == candidate_expr)

                result = solver.check()
                if result == z3.sat:
                    model = solver.model()
                    counterexample = {str(var): model[var] for var in model.decls() if str(var) in problem.z3variables}

                    print('Counterexample:', counterexample)

                    solver.pop()

                    additional_constraints = []
                    for var_name, z3_var in problem.z3variables.items():
                        if var_name in counterexample:
                            additional_constraint = z3_var != counterexample[var_name]
                            additional_constraints.append(additional_constraint)

                    solver.add(additional_constraints)
                    break
                else:
                    print('Valid candidate found:', candidate_expr)
                    solver.pop()
                    return candidate_expr

        depth += 1

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
