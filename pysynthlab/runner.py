from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, FileType
import cvc5
import z3

from pysynthlab.helpers.parser.src.ast import CommandKind, ASTVisitor
from pysynthlab.synthesis_problem import SynthesisProblem

def main(args):
    file = args.input_file.read()

    # Testing smt input
    # solver = z3.Solver()
    # solver.add(z3.parse_smt2_string(file))
    # solver.check()
    # model = solver.model()
    # print('Model:', model)

    problem = SynthesisProblem(file, int(args.sygus_standard))
    problem.info()
    print(problem.get_logic())
    smt_lib_problem = translate_to_smt_lib_2(problem.__str__())

    solver = z3.Solver()

    for command in smt_lib_problem.split('\n'):
        if command.strip() != '':
            solver.add(z3.parse_smt2_string(command))

    result = solver.check()

    if result == z3.sat:
        print('Satisfiable!')
        model = solver.model()
        print('Model:', model)
    else:
        print('Unsatisfiable!')


def translate_to_smt_lib_2(sygus_content):
    smt_lib_2_content = []
    same_commands = {
        'declare-datatype',
        'declare-datatypes',
        'declare-sort',
        'define-fun',
        'define-sort',
        'set-info',
        'set-logic',
        'set-option'
    }
    for line in sygus_content.split('\n'):
        line = line.strip()
        if not line or line.startswith(';'):
            continue

        tokens = line.replace('(', ' ( ').replace(')', ' ) ').split()

        command = tokens[1]
        if command in same_commands:  # Commands that are the same in SyGuS-IF and SMT-LIB2
            smt_lib_2_content.append(line)
        elif command == 'synth-fun':
            function_symbol = tokens[2]
            variable_sorts = ' '.join(f'({var})' for var in tokens[3:-2:2])
            return_sort = tokens[-2]
            smt_lib_2_content.append(f'(declare-fun {function_symbol} ({variable_sorts}) {return_sort})')
        elif command == 'check-synth':
            smt_lib_2_content.append('(check-sat)')
        elif command == 'assume':
            term = ' '.join(tokens[2:-1])
            smt_lib_2_content.append(f'(assert {term})')
        elif command == 'declare-var':
            symbol = tokens[2]
            sort = tokens[3]
            smt_lib_2_content.append(f'(declare-fun {symbol} () {sort})')
        elif command == 'constraint':
            term = ' '.join(tokens[2:-1])
            smt_lib_2_content.append(f'(assert {term})')
        elif command == '(declare-weight':
            symbol = tokens[2]
            attributes = ' '.join(tokens[3:])
            smt_lib_2_content.append(f'; (declare-weight {symbol} {attributes})')
        else:
            smt_lib_2_content.append(line)

    smt_lib_2_content.append('(get-model)')
    return '\n'.join(smt_lib_2_content)


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
