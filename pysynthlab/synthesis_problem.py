import itertools

import z3
import pyparsing
from pysynthlab.helpers.parser.src import symbol_table_builder
from pysynthlab.helpers.parser.src.ast import Program, CommandKind, GrammarTermKind
from pysynthlab.helpers.parser.src.resolution import SymbolTable, FunctionKind, SortDescriptor
from pysynthlab.helpers.parser.src.v1.parser import SygusV1Parser
from pysynthlab.helpers.parser.src.v1.printer import SygusV1ASTPrinter
from pysynthlab.helpers.parser.src.v2.parser import SygusV2Parser
from pysynthlab.helpers.parser.src.v2.printer import SygusV2ASTPrinter


class SynthesisProblem:
    MIN_CONST = -10
    MAX_CONST = 10

    def __init__(self, problem: str, solver: z3.Solver, sygus_standard: int = 1, options: object = None):

        if options is None:
            options = {}
        self.options: object = options
        self.sygus_standard: int = sygus_standard
        self.parser: SygusV1Parser | SygusV2Parser = SygusV2Parser() if sygus_standard == 2 else SygusV1Parser()
        self.input_problem: str = problem
        self.problem: Program = self.parser.parse(problem)
        self.symbol_table: SymbolTable = symbol_table_builder.SymbolTableBuilder.run(self.problem)
        self.printer: SygusV2ASTPrinter | SygusV1ASTPrinter = SygusV2ASTPrinter(self.symbol_table) \
            if sygus_standard == 2 \
            else SygusV1ASTPrinter(self.symbol_table, options)
        self.solver = solver
        self.commands = [x for x in self.problem.commands]
        self.constraints = [x for x in self.problem.commands if x.command_kind == CommandKind.CONSTRAINT]

        self.z3variables = {}
        self.z3function_definitions = []

        pyparsing.ParserElement.enablePackrat()
        self.smt_problem = self.convert_sygus_to_smt()

        self.initialise_variables()
        self.z3functions = {}
        self.initialise_z3_functions()
        self.operand_pool = []
        self.initialize_operand_pool()
        self.additional_constraints = []

        self.func_name, self.z3_func = list(self.z3functions.items())[0]  # todo: refactor for problems with more than one
        self.func = self.get_synth_func(self.func_name)
        self.arg_sorts = [self.convert_sort_descriptor_to_z3_sort(sort_descriptor) for sort_descriptor in
                          self.func.argument_sorts]
        self.func_args = [z3.Const(name, sort) for name, sort in zip(self.func.argument_names, self.arg_sorts)]

    def __str__(self) -> str:
        return self.printer.run(self.problem, self.symbol_table)

    def info(self) -> None:
        print(self)

    def convert_sygus_to_smt(self):
        i_expr = pyparsing.QuotedString(quoteChar='"') | pyparsing.QuotedString(quoteChar='|', unquoteResults=False)
        s_expr = pyparsing.nestedExpr(opener='(', closer=')', ignoreExpr=i_expr)
        s_expr.ignore(';' + pyparsing.restOfLine)

        sygus_parser = pyparsing.ZeroOrMore(s_expr)
        ast = sygus_parser.parseString(self.input_problem, parseAll=True).asList()

        for statement in ast:
            if statement[0] == 'constraint':
                statement[0] = 'assert'
            elif statement[0] == 'check-synth':
                statement[0] = 'check-sat'
            elif statement[0] == 'synth-fun':
                statement[0] = 'declare-fun'
                statement[2] = [var_decl[1] for var_decl in statement[2]]

        def serialise(line):
            return line if type(line) is not list else f'({" ".join(serialise(expression) for expression in line)})'

        return '\n'.join(serialise(statement) for statement in ast)

    def get_logic(self):
        return self.symbol_table.logic_name

    def get_synth_funcs(self):
        return self.symbol_table.synth_functions

    def get_synth_func(self, symbol):
        return next(filter(lambda x:
                           x.function_kind == FunctionKind.SYNTH_FUN and x.identifier.symbol == symbol,
                           list(self.symbol_table.synth_functions.values())))

    def get_var_symbols(self):
        return [x.symbol for x in self.problem.commands if x.command_kind == CommandKind.DECLARE_VAR]

    def get_function_symbols(self):
        return [x.symbol for x in self.problem.commands if x.command_kind == CommandKind.DECLARE_FUN]

    def extract_synth_function(self, function_symbol) -> str:
        synthesis_function = self.get_synth_func(function_symbol)
        func_problem = next(filter(lambda x:
                                   x.command_kind == CommandKind.SYNTH_FUN and x.function_symbol == function_symbol,
                                   self.problem.commands))

        arg_sorts = [str(arg_sort.identifier) for arg_sort in synthesis_function.argument_sorts]

        return f'(declare-fun {function_symbol} ({" ".join(arg_sorts)}) {func_problem.range_sort_expression.identifier.symbol})'

    def initialise_variables(self):
        for variable in self.problem.commands:
            if variable.command_kind == CommandKind.DECLARE_VAR and variable.sort_expression.identifier.symbol == 'Int':
                z3_var = z3.Int(variable.symbol, self.solver.ctx)
                self.z3variables[variable.symbol] = z3_var

    def initialise_z3_functions(self):
        for func in self.get_synth_funcs().values():
            z3_arg_sorts = [self.convert_sort_descriptor_to_z3_sort(s) for s in func.argument_sorts]
            z3_range_sort = self.convert_sort_descriptor_to_z3_sort(func.range_sort)
            self.z3functions[func.identifier.symbol] = z3.Function(func.identifier.symbol, *z3_arg_sorts, z3_range_sort)

    @staticmethod
    def convert_sort_descriptor_to_z3_sort(sort_descriptor: SortDescriptor):
        sort_symbol = sort_descriptor.identifier.symbol
        return {
            'Int': z3.IntSort(),
            'Bool': z3.BoolSort(),
        }.get(sort_symbol, None)

    def generate_linear_integer_expressions(self, depth, size_limit=10, current_size=0):
        # if the current depth == 0 or current_size exceeds the limit, yield integer values and variables
        if depth == 0 or current_size >= size_limit:
            yield from [z3.IntVal(i) for i in range(self.MIN_CONST, self.MAX_CONST)] + list(self.z3variables.values())
        else:
            # for each variable, generate expressions by combining with other expressions of lesser depth
            for var_name, var in self.z3variables.items():
                # if the current size is within the limit, just yield the variable
                if current_size < size_limit:
                    yield var
                for expr in self.generate_linear_integer_expressions(depth - 1, size_limit, current_size + 1):
                    if current_size + 1 < size_limit and not z3.eq(expr, var):
                        yield var + expr
                        yield var - expr
                        yield expr - var

                    # conditional expressions also constrained to the size limit
                    for other_expr in self.generate_linear_integer_expressions(depth - 1, size_limit, current_size + 1):
                        if current_size + 2 < size_limit:
                            yield z3.If(var > other_expr, var, other_expr)
                            yield z3.If(var < other_expr, var, other_expr)
                            yield z3.If(var == other_expr, var, expr)
                            yield z3.If(var != other_expr, var, expr)

    def check_counterexample(self, model):
        for constraint in self.solver.assertions():
            if not model.eval(constraint, model_completion=True):
                return {str(arg): model[arg] for arg in self.func_args}
        return None

    def get_additional_constraints(self, counterexample):
        return [var != counterexample[var] for var in self.func_args]

    def generate_candidate_expression(self):
        expressions = self.generate_linear_integer_expressions(depth=0)
        for expr in itertools.islice(expressions, 200):  # limit breadth
            return expr

    ## Fast enumerative synth

    def enumerate_expressions(self, depth):
        if depth == 0:
            return self.z3variables.values() + [z3.IntVal(i) for i in range(self.MIN_CONST, self.MAX_CONST)]
        else:
            prev_level_expressions = self.enumerate_expressions(depth - 1)
            new_expressions = []

            for expr1 in prev_level_expressions:
                for expr2 in prev_level_expressions:
                    if isinstance(expr1, z3.ArithRef) and isinstance(expr2, z3.ArithRef):
                        new_expressions.append(expr1 + expr2)
                        new_expressions.append(expr1 - expr2)
                        new_expressions.append(expr1 * expr2)

                    new_expressions.append(z3.If(expr1 > expr2, expr1, expr2))
                    new_expressions.append(z3.If(expr1 < expr2, expr1, expr2))
                    new_expressions.append(z3.If(expr1 == expr2, expr1, expr2))
                    new_expressions.append(z3.If(expr1 != expr2, expr1, expr2))
            return new_expressions

    def get_grammar(self):
        grammars = {}
        for synth_func_cmd in self.get_synth_funcs().values():
            synth_grammar = synth_func_cmd.synthesis_grammar
            if synth_grammar is None:
                continue

            output_sort = self.convert_sort_descriptor_to_z3_sort(synth_func_cmd.range_sort)

            if output_sort not in grammars:
                grammars[output_sort] = {'nonterminals': [], 'rules': {}}

            for nt_symbol, nt_sort in synth_grammar.nonterminals:
                nt_sort_z3 = self.convert_sort_descriptor_to_z3_sort(nt_sort)
                grammars[output_sort]['nonterminals'].append((nt_symbol, nt_sort_z3))
                if nt_symbol in synth_grammar.grouped_rule_lists:
                    grouped_rule_list = synth_grammar.grouped_rule_lists[nt_symbol]
                    grammars[output_sort]['rules'][nt_symbol] = []
                    for expansion_rule in grouped_rule_list.expansion_rules:
                        processed_rule = self.process_expansion_rule(expansion_rule)
                        grammars[output_sort]['rules'][nt_symbol].append(processed_rule)

        return grammars

    def process_expansion_rule(self, grammar_term):
        expansions = []
        if grammar_term.grammar_term_kind == GrammarTermKind.CONSTANT:
            expansions.extend(self.handle_constant_expansion())
        elif grammar_term.grammar_term_kind == GrammarTermKind.VARIABLE:
            expansions.extend(self.handle_variable_expansion())
        elif grammar_term.grammar_term_kind == GrammarTermKind.BINDER_FREE:
            expansions.extend(self.handle_binder_free_expansion())
        return expansions

    def handle_constant_expansion(self):
        constant_terms = []
        for const in range(self.MIN_CONST, self.MAX_CONST + 1):
            constant_terms.append(z3.IntVal(const))

        return constant_terms

    def handle_variable_expansion(self):
        return []

    def handle_binder_free_expansion(self):
        arithmetic_terms = []
        for op in ['+', '-', '*', '/']:
            for operand1 in self.operand_pool:
                for operand2 in self.operand_pool:
                    if operand2 != operand1:  # avoid expressions like x - x
                        term = self.create_arithmetic_term(op, operand1, operand2)
                        if term is not None:
                            arithmetic_terms.append(term)
        return arithmetic_terms

    def initialize_operand_pool(self):
        self.operand_pool = list(self.z3variables.values())
        integer_constants = [z3.IntVal(i) for i in range(self.MIN_CONST, self.MAX_CONST)]
        self.operand_pool.extend(integer_constants)

    def create_arithmetic_term(self, op, operand1, operand2):
        if op == '+':
            return operand1 + operand2
        elif op == '-':
            return operand1 - operand2
        elif op == '*':
            return operand1 * operand2