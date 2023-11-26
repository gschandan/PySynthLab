from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, FileType
import cvc5

from pysynthlab.helpers.parser.src.ast import CommandKind, ASTVisitor
from pysynthlab.synthesis_problem import SynthesisProblem

class CVC5DeclarationVisitor(ASTVisitor):
    def __init__(self, solver):
        super().__init__('CVC5DeclarationVisitor')
        self.solver = solver

    def visit_declare_var_command(self, declare_var_command):
        symbol = declare_var_command.symbol
        sort_expr = declare_var_command.sort_expression

        cvc5_sort = self.convert_sort_expr(sort_expr)

        cvc5_var = self.solver.mkVar(symbol, cvc5_sort)

        print(f"Declared variable: {symbol} of sort {sort_expr}")

    def convert_sort_expr(self, sort_expr):
        cvc5_sort = self.cvc5_sort_converter(sort_expr)
        return cvc5_sort

    def cvc5_sort_converter(self, sort_expr):
        if sort_expr.identifier.symbol == 'Int':
            return cvc5.Sort(cvc5.SortKind.INTEGER)
        elif sort_expr.identifier.symbol == 'Bool':
            return cvc5.Sort(cvc5.SortKind.BOOLEAN)
        else:
            raise ValueError(f"Unsupported sort: {sort_expr}")


def main(args):
    file = args.input_file.read()
    problem = SynthesisProblem(file, int(args.sygus_standard))
    problem.info()
    print(problem.get_logic())

    solver = cvc5.Solver()

    # required options

    solver.setOption("sygus", "true")
    solver.setOption("incremental", "false")

    # set the logic
    solver.setLogic(problem.get_logic())

    declaration_visitor = CVC5DeclarationVisitor(solver)

    for command in problem.problem.commands:
        if command.command_kind == CommandKind.DECLARE_VAR:
            command.accept(declaration_visitor)

    def str_to_term(str_input):
        def get_kind(sym):
            if sym == "+":
                return cvc5.Kind.PLUS
            elif sym == "-":
                return cvc5.Kind.MINUS
            elif sym == "*":
                return cvc5.Kind.MULT
            elif sym == "/":
                return cvc5.Kind.DIVISION
            elif sym == "=":
                return cvc5.Kind.EQUAL

        # Tokenize the input
        tokens = str_input.split()

        stack = []
        for t in tokens:
            if t in ["+", "-", "*", "/", "="]:
                # Handle operators
                right = stack.pop()
                left = stack.pop()
                stack.append(solver.mkTerm(get_kind(t), left, right))
            else:
                # Handle variables or constants
                if t.isdigit():
                    # If it's a number, create a constant
                    stack.append(solver.mkConst(solver.getIntegerSort(), t))
                else:
                    # Assume it's a variable
                    stack.append(solver.mkVar(t, solver.getIntegerSort()))

        return stack.pop()

    solver.assertFormula(problem.__str__())

    result = solver.checkSynth()


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '-b', '--binarize', action='store_true',
        help='Convert all chainable operators to binary operator applications')
    parser.add_argument(
        '-q', '--quiet', action='store_true',
        help='Suppress all messages and debugging output')
    parser.add_argument(
        '-u', '--no-unary-minus', action='store_true',
        help='Convert all (- x) terms to (- 0 x)')

    parser.add_argument(
        '-s', '--sygus-standard', default='2', choices=['1','2'],
        help='The SyGuS language standard used in the input file')

    parser.add_argument(
        'input_file', type=FileType('r'),
        help='Path to an input file (or stdin if "-")')

    main(parser.parse_args())
