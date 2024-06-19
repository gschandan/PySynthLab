import argparse
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, FileType
from typing import List, Dict, Callable

from z3 import z3

from src.cegis.z3.fast_enumerative_synthesis import FastEnumerativeSynthesis
from src.cegis.z3.synthesis_problem_z3 import SynthesisProblemOptions, SynthesisProblem


def main(args: argparse.Namespace) -> None:
    """
    Main function to run the synthesis problem.

    :param args: Command-line arguments. 
    """

    file_content = args.input_file.read()

    options = SynthesisProblemOptions(
        sygus_standard=int(args.sygus_standard),
        verbose=args.verbose
    )

    # problem = SynthesisProblem(file_content, options)
    # 
    # if options.verbose < 2:
    #     problem.info_sygus()
    #     problem.info_smt()
    # 
    # problem.execute_cegis()
    
    def create_candidate_function(candidate_expr: z3.ExprRef, arg_sorts: List[z3.SortRef]) -> Callable:
        args = [z3.Var(i, sort) for i, sort in enumerate(arg_sorts)]

        def candidate_function(*values):
            if len(values) != len(args):
                raise ValueError("Incorrect number of arguments.")

            simplified_expr = z3.simplify(
                z3.substitute(candidate_expr, [(arg, value) for arg, value in zip(args, values)]))
            return simplified_expr
        
        candidate_function.__name__ = candidate_expr.sexpr()
        return candidate_function
    
    problem = FastEnumerativeSynthesis(file_content, options)

    starting_depth = 0
    max_depth = 3
    for depth in range(starting_depth, max_depth + 1):
        generated_terms = problem.generate(starting_depth, depth)
        for func_name, func in problem.context.z3_synth_functions.items():
            func_str = f"{func_name}({', '.join(problem.context.z3_synth_function_args[func_name].keys())})"
            candidate_functions = generated_terms[func.range()][depth-starting_depth]
            for candidate_function in candidate_functions:
                candidate_functions_callable = create_candidate_function(candidate_function, [x.sort() for x in list(problem.context.variable_mapping_dict[func_name].keys())] )   
                candidate = candidate_functions_callable(*list(problem.context.z3_synth_function_args[func_name].values()))
                result = problem.test_candidates([func_str], [candidate])
                if result:
                    problem.print_msg(f"Found solution for function {func_name}: {candidate_functions[0]}", level=0)
                    return
        starting_depth += 1
    problem.print_msg(f"No solution found up to depth {max_depth}", level=0)
    

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '-v', '--verbose', type=int, default=0, choices=[0, 1, 2],
        help='Verbosity level:\n'
             '\t 0 = no suppression; all output printed to console\n'
             '\t 1 = suppress warnings\n'
             '\t 2 = suppress all output except success/failure')

    parser.add_argument(
        '-s', '--sygus-standard', default='2', choices=['1', '2'],
        help='The SyGuS language standard used in the input file')

    parser.add_argument(
        'input_file', type=FileType('r'),
        help='Path to an input file (or stdin if "-")')

    main(parser.parse_args())
