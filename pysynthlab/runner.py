import argparse
import typing
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, FileType
from typing import Collection

from pysynthlab.cegis.z3.synthesis_problem_z3 import SynthesisProblemOptions, SynthesisProblem
from z3 import *


def main(args: argparse.Namespace) -> None:
    """
    Main function to run the synthesis problem.

    :param args: Command-line arguments.
    """

    def make_expression(possible_expression):
        if is_expr(possible_expression):
            return possible_expression
        elif isinstance(possible_expression, int):
            return IntVal(possible_expression)
        else:
            raise Exception("Cannot convert: %s" % possible_expression.__repr__())

    def substitute_constraints(constraints: Collection[z3.ExprRef], func: z3.FuncDeclRef,
                               candidate_expression: typing.Union[z3.FuncDeclRef, z3.QuantifierRef, typing.Callable, int, z3.IntNumRef]) -> typing.List[z3.ExprRef]:
        def reconstruct_expression(expr: z3.ExprRef) -> z3.ExprRef:
            if is_app(expr) and expr.decl() == func:
                new_args = [reconstruct_expression(arg) for arg in expr.children()]
                if callable(candidate_expression):
                    return candidate_expression(*new_args)
                elif isinstance(candidate_expression, (FuncDeclRef, QuantifierRef)):
                    return candidate_expression(*new_args)
                else:
                    return make_expression(candidate_expression)
            elif is_app(expr):
                new_args = [reconstruct_expression(arg) for arg in expr.children()]
                return expr.decl()(*new_args)
            else:
                return expr

        if isinstance(candidate_expression, (int, z3.IntNumRef)):
            substitution_pairs = [(func, make_expression(candidate_expression))]
        elif callable(candidate_expression):
            substitution_pairs = [(func, lambda *args: candidate_expression(*[make_expression(arg) for arg in args]))]
        else:
            substitution_pairs = [(func, candidate_expression)]
    
        return [substitute(c, *substitution_pairs) for c in constraints]
    
    x, y = Ints('x y')
    expr = x + y
    
    def functionFromExpr(args, expr):
        return lambda *fargs: substitute(expr, *zip(args, [make_expression(e) for e in fargs]))
    
    f = functionFromExpr([x, y], expr)
    # These work ok
    print(f(x, y))          # Should print: x + y
    print(f(x, IntVal(4)))  # Should print: x + 4
    print(f(IntVal(3), IntVal(4)))  # Should print: 3 + 4
    
    # This does not work
    x, y = Ints('x y')
    f = Function('f', IntSort(), IntSort(), IntSort())
    
    commutativity_constraint = f(x, y) == f(y, x)
    print("Original commutativity constraint:")
    print(commutativity_constraint)
    def func_2(a, b):
        return If(a >= b, a, b)
    
    substituted_constraints = substitute_constraints([commutativity_constraint], f, func_2)
    for constraint in substituted_constraints:
        print(constraint)
    
    # Test substitution with an integer constant
    substituted_constraints = substitute_constraints([commutativity_constraint], f, 42)
    print("\nSubstituted constraints with integer constant:")
    for constraint in substituted_constraints:
        print(constraint)
        
    # file_content = args.input_file.read()
    # 
    # options = SynthesisProblemOptions(
    #     sygus_standard=int(args.sygus_standard),
    #     verbose=args.verbose
    # )
    # 
    # problem = SynthesisProblem(file_content, options)
    # 
    # if options.verbose < 2:
    #     problem.info()
    # 
    # problem.execute_cegis()


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
