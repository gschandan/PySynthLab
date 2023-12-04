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
    smt_lib_problem = translate_to_smt_lib_2(problem)

    solver = z3.Solver()

    for command in smt_lib_problem.split('\n'):
        if command.strip() != '':
            print(f"Parsed command: {command}")
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

    for line in str(sygus_content).__str__().split('\n'):
        line = line.strip()
        if not line or line.startswith(';'):
            continue

        tokens = line.replace('(', ' ( ').replace(')', ' ) ').split()

        command = tokens[1]

        if command in same_commands:
            smt_lib_2_content.append(line)
        elif command == 'synth-fun':
            smt_lib_2_content.append(extract_synth_function(sygus_content, tokens[2]))
        elif command == 'assume':
            term = ' '.join(tokens[2:-1])
            smt_lib_2_content.append(f'(assert {term})')
        elif command == 'declare-var':
            symbol = tokens[2]
            sort = tokens[3]
            smt_lib_2_content.append(f'(declare-fun {symbol} () {sort})')
        elif command == 'constraint':
            term = ''.join([' (' if s == '(' else s for s in tokens[2:-1]])
            smt_lib_2_content.append(f'(assert{term})')
        elif command == 'declare-weight':
            symbol = tokens[2]
            attributes = ' '.join(tokens[3:])
            smt_lib_2_content.append(f'; (declare-weight {symbol} {attributes})')
        elif command == 'check-synth':
            pass

    smt_lib_2_content.append('(check-sat)')
    smt_lib_2_content.append('(get-model)')
    return '\n'.join(smt_lib_2_content)


def extract_synth_function(sygus_content, function_symbol) -> str:
    synthesis_function = sygus_content.get_synth_func(function_symbol)
    func_problem = next(filter(lambda x:
                       x.command_kind == CommandKind.SYNTH_FUN and x.function_symbol == function_symbol,
                       sygus_content.problem.commands))

    arg_sorts = [str(arg_sort.identifier) for arg_sort in synthesis_function.argument_sorts]

    return f'(declare-fun {function_symbol} ({" ".join(arg_sorts)}) {func_problem.range_sort_expression.identifier.symbol})'


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
        '-s', '--sygus-standard', default='2', choices=['1', '2'],
        help='The SyGuS language standard used in the input file')

    parser.add_argument(
        'input_file', type=FileType('r'),
        help='Path to an input file (or stdin if "-")')

    main(parser.parse_args())
