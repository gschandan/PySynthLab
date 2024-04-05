import itertools

import cvc5
from cvc5 import Kind

import pyparsing
from pysynthlab.helpers.parser.src import symbol_table_builder, ast
from pysynthlab.helpers.parser.src.ast import Program, CommandKind, GrammarTermKind, ConstraintCommand, IdentifierTerm, \
    LiteralTerm, LiteralKind, FunctionApplicationTerm, QuantifiedTerm, QuantifierKind
from pysynthlab.helpers.parser.src.resolution import SymbolTable, FunctionKind, SortDescriptor
from pysynthlab.helpers.parser.src.v1.parser import SygusV1Parser
from pysynthlab.helpers.parser.src.v1.printer import SygusV1ASTPrinter
from pysynthlab.helpers.parser.src.v2.parser import SygusV2Parser
from pysynthlab.helpers.parser.src.v2.printer import SygusV2ASTPrinter


class SynthesisProblemCvc5:
    MIN_CONST = -2
    MAX_CONST = 2

    def __init__(self, problem: str, sygus_standard: int = 1, options: object = None):
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

        self.enumerator_solver = cvc5.Solver()
        self.verification_solver = cvc5.Solver()

        self.commands = [x for x in self.problem.commands]
        self.constraints = [x for x in self.problem.commands if x.command_kind == CommandKind.CONSTRAINT]
        self.smt_problem = self.convert_sygus_to_smt()
        self.synthesis_functions = []

        self.cvc5variables = {}
        self.cvc5_synth_functions = {}
        self.cvc5_predefined_functions = {}
        self.cvc5_constraints = []

        self.initialise_cvc5_variables()
        self.initialise_cvc5_synth_functions()
        self.initialise_cvc5_predefined_functions()

        self.assertions = set()
        self.counterexamples = set()
        self.negated_assertions = set()
        self.additional_constraints = []
        self.original_assertions = []

        # todo: refactor for problems with more than one func to synthesize
        self.func_name, self.cvc5_func = list(self.cvc5_synth_functions.items())[0]
        self.func_to_synthesise = self.get_synth_func(self.func_name)
        self.func_args = [self.enumerator_solver.mkConst(self.enumerator_solver.getIntegerSort(), name) for name in
                          self.func_to_synthesise.argument_names]
        self.arg_sorts = [self.convert_sort_descriptor_to_cvc5_sort(sort_descriptor) for sort_descriptor in
                          self.func_to_synthesise.argument_sorts]

        self.parse_constraints()

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

        constraints = []
        constraint_indices = []
        filtered_ast = []

        for i, statement in enumerate(ast):
            if statement[0] == 'constraint':
                constraints.append(statement[1])
                constraint_indices.append(len(filtered_ast))
                filtered_ast.append(statement)
            elif statement[0] == 'check-synth':
                statement[0] = 'check-sat'
                filtered_ast.append(statement)
            elif statement[0] != 'synth-fun':
                filtered_ast.append(statement)

        if constraints:
            conjoined_constraints = ['and'] + constraints
            filtered_ast[constraint_indices[0]] = ['assert', conjoined_constraints]
            for index in reversed(constraint_indices[1:]):
                del filtered_ast[index]

        def serialise(line):
            return line if type(line) is not list else f'({" ".join(serialise(expression) for expression in line)})'

        return '\n'.join(serialise(statement) for statement in filtered_ast)

    def get_logic(self):
        return self.symbol_table.logic_name

    def get_synth_funcs(self):
        return self.symbol_table.synth_functions

    def get_predefined_funcs(self):
        return self.symbol_table.user_defined_functions

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

    def initialise_cvc5_variables(self):
        for variable in self.problem.commands:
            if variable.command_kind == CommandKind.DECLARE_VAR and variable.sort_expression.identifier.symbol == 'Int':
                cvc5_var = self.enumerator_solver.mkConst(self.enumerator_solver.getIntegerSort(), variable.symbol)
                self.cvc5variables[variable.symbol] = cvc5_var

    def initialise_cvc5_synth_functions(self):
        for func in self.get_synth_funcs().values():
            cvc5_arg_sorts = [self.convert_sort_descriptor_to_cvc5_sort(s) for s in func.argument_sorts]
            cvc5_range_sort = self.convert_sort_descriptor_to_cvc5_sort(func.range_sort)
            self.cvc5_synth_functions[func.identifier.symbol] = self.enumerator_solver.mkConst(cvc5_range_sort,func.identifier.symbol)

    def initialise_cvc5_predefined_functions(self):
        for func in self.get_predefined_funcs().values():
            cvc5_arg_sorts = [self.convert_sort_descriptor_to_cvc5_sort(s) for s in func.argument_sorts]
            cvc5_range_sort = self.convert_sort_descriptor_to_cvc5_sort(func.range_sort)
            self.cvc5_predefined_functions[func.identifier.symbol] = self.enumerator_solver.mkConst(cvc5_range_sort,func.identifier.symbol)

    def parse_constraints(self):
        for constraint in self.constraints:
            if isinstance(constraint, ast.ConstraintCommand):
                term = self.parse_term(constraint.constraint)
                self.cvc5_constraints.append(term)

    def parse_term(self, term: ast.Term) -> cvc5.Term:
        if isinstance(term, ast.IdentifierTerm):
            symbol = term.identifier.symbol
            if symbol in self.cvc5variables:
                return self.cvc5variables[symbol]
            elif symbol in self.cvc5_synth_functions:
                return self.cvc5_synth_functions[symbol]
            elif symbol in self.cvc5_predefined_functions:
                return self.cvc5_predefined_functions[symbol]
            else:
                raise ValueError(f"Undefined symbol: {symbol}")
        elif isinstance(term, ast.LiteralTerm):
            literal = term.literal
            if literal.literal_kind == ast.LiteralKind.NUMERAL:
                return self.enumerator_solver.mkInteger(int(literal.literal_value))
            elif literal.literal_kind == ast.LiteralKind.BOOLEAN:
                return self.enumerator_solver.mkBoolean(literal.literal_value.lower() == "true")
            else:
                raise ValueError(f"Unsupported literal kind: {literal.literal_kind}")
        elif isinstance(term, ast.FunctionApplicationTerm):
            func_symbol = term.function_identifier.symbol
            if func_symbol == "=":
                assert len(term.arguments) == 2, " == should have 2 arguments"
                lhs = self.parse_term(term.arguments[0])
                rhs = self.parse_term(term.arguments[1])
                return self.enumerator_solver.mkTerm(Kind.EQUAL, lhs, rhs)
            elif func_symbol in self.cvc5_synth_functions:
                args = [self.parse_term(arg) for arg in term.arguments]
                return self.enumerator_solver.mkTerm(Kind.APPLY_UF, self.enumerator_solver.mkConst(
                    self.enumerator_solver.getBooleanSort(), func_symbol), *args)
            elif func_symbol in self.cvc5_predefined_functions:
                func = self.cvc5_predefined_functions[func_symbol]
                args = [self.parse_term(arg) for arg in term.arguments]
                return self.enumerator_solver.mkTerm(func.getKind(), *args)
            else:
                raise ValueError(f"Undefined function symbol: {func_symbol}")
        elif isinstance(term, ast.QuantifiedTerm):
            variables = [(v.symbol, self.enumerator_solver.mkConst(self.enumerator_solver.getIntegerSort(), v.symbol))
                         for v in term.quantified_variables]
            body = self.parse_term(term.term_body)
            if term.quantifier_kind == ast.QuantifierKind.FORALL:
                return self.enumerator_solver.mkTerm(Kind.FORALL, [v[1] for v in variables], body)
            elif term.quantifier_kind == ast.QuantifierKind.EXISTS:
                return self.enumerator_solver.mkTerm(Kind.EXISTS, [v[1] for v in variables], body)
            else:
                raise ValueError(f"Unsupported quantifier kind: {term.quantifier_kind}")
        else:
            raise ValueError(f"Unsupported term type: {type(term)}")

    def generate_linear_integer_expressions(self, depth, size_limit=6, current_size=0):
        if depth == 0 or current_size >= size_limit:
            yield from [self.enumerator_solver.mkInteger(i) for i in range(self.MIN_CONST, self.MAX_CONST + 1)] + list(self.cvc5variables.values())
            return

        for var in self.cvc5variables.values():
            if current_size < size_limit:
                yield var
                yield self.enumerator_solver.mkTerm(Kind.NEG, var)

            for expr in self.generate_linear_integer_expressions(depth - 1, size_limit, current_size + 1):
                yield self.enumerator_solver.mkTerm(Kind.ADD, var, expr)
                yield self.enumerator_solver.mkTerm(Kind.SUB, var, expr)
                yield self.enumerator_solver.mkTerm(Kind.SUB, expr, var)

            for expr in self.generate_linear_integer_expressions(depth - 1, size_limit, current_size + 2):
                if current_size + 3 <= size_limit:
                    yield self.enumerator_solver.mkTerm(Kind.ITE, self.enumerator_solver.mkTerm(Kind.GT, var, expr),var, expr)
                    yield self.enumerator_solver.mkTerm(Kind.ITE, self.enumerator_solver.mkTerm(Kind.LT, var, expr),var, expr)
                    yield self.enumerator_solver.mkTerm(Kind.ITE, self.enumerator_solver.mkTerm(Kind.NEQ, var, expr),var, expr)

    def generate_candidate_functions(self, depth, size_limit=6, current_size=0):
        if depth == 0 or current_size >= size_limit:
            for i in range(self.MIN_CONST, self.MAX_CONST + 1):
                def const_func(*args):
                    return self.enumerator_solver.mkInteger(i)
                yield const_func

            for var in self.cvc5variables.keys():
                def var_func(*args):
                    return var
                yield var_func
            return

        for var_name, var in self.cvc5variables.items():
            if current_size < size_limit:
                def identity_func(*args):
                    return args[list(self.cvc5variables.keys()).index(var_name)]
                yield identity_func

                def neg_func(*args):
                    return self.enumerator_solver.mkTerm(Kind.NEG,args[list(self.cvc5variables.keys()).index(var_name)])
                yield neg_func

            for func in self.generate_candidate_functions(depth - 1, size_limit, current_size + 1):
                def add_func(*args):
                    return self.enumerator_solver.mkTerm(Kind.ADD,args[list(self.cvc5variables.keys()).index(var_name)], func(*args))
                yield add_func

                def sub_func(*args):
                    return self.enumerator_solver.mkTerm(Kind.SUB,args[list(self.cvc5variables.keys()).index(var_name)],func(*args))
                yield sub_func

                def sub_func_rev(*args):
                    return self.enumerator_solver.mkTerm(Kind.SUB, func(*args), args[list(self.cvc5variables.keys()).index(var_name)])
                yield sub_func_rev

            for func in self.generate_candidate_functions(depth - 1, size_limit, current_size + 2):
                if current_size + 3 <= size_limit:
                    def ite_gt_func(*args):
                        return self.enumerator_solver.mkTerm(Kind.ITE,self.enumerator_solver.mkTerm(Kind.GT, args[list(self.cvc5variables.keys()).index(var_name)],func(*args)),
                                                             args[list(self.cvc5variables.keys()).index(var_name)],func(*args))
                    yield ite_gt_func

                    def ite_lt_func(*args):
                        return self.enumerator_solver.mkTerm(Kind.ITE,self.enumerator_solver.mkTerm(Kind.LT, args[list(self.cvc5variables.keys()).index(var_name)],func(*args)),
                                                             args[list(self.cvc5variables.keys()).index(var_name)],func(*args))
                    yield ite_lt_func

                    def ite_neq_func(*args):
                        return self.enumerator_solver.mkTerm(Kind.ITE, self.enumerator_solver.mkTerm(Kind.NEQ, args[list(self.cvc5variables.keys()).index(var_name)],func(*args)),
                                                             args[list(self.cvc5variables.keys()).index(var_name)],func(*args))
                    yield ite_neq_func

    def check_counterexample(self, model):
        for constraint in self.original_assertions:
            if not model.eval(constraint, model_completion=True).getBooleanValue():
                return {str(arg): model.eval(arg, model_completion=True).getIntegerValue() for arg in self.func_args}
        return None

    def get_additional_constraints(self, counterexample):
        constraints = [self.enumerator_solver.mkTerm(Kind.NEQ, var,
                                                     self.enumerator_solver.mkInteger(counterexample[var.__str__()]))
                       for var in self.func_args]
        return self.enumerator_solver.mkTerm(Kind.AND, *constraints)

    def generate_candidate_expression(self, depth=0):
        expressions = self.generate_linear_integer_expressions(depth)
        for expr in itertools.islice(expressions, 200):  # limit breadth
            return expr

    def convert_sort_descriptor_to_cvc5_sort(self, sort_descriptor: SortDescriptor):
        sort_symbol = sort_descriptor.identifier.symbol
        return {
            'Int': self.enumerator_solver.getIntegerSort(),
            'Bool': self.enumerator_solver.getBooleanSort(),
        }.get(sort_symbol, None)

    @staticmethod
    def negate_assertions(assertions):
        negated_assertions = []
        for assertion in assertions:
            args = assertion.getNumChildren()
            if assertion.getKind() in [Kind.AND, Kind.OR, Kind.NOT]:
                if args > 1:
                    negated_children = [assertion.notTerm(assertion.getChild(i)) for i in range(args)]
                    negated_assertions.append(assertion.orTerm(*negated_children))
                else:
                    negated_assertions.append(assertion.notTerm(assertion))
            elif assertion.getKind() == Kind.APPLY_UF and args == 2:
                if assertion.getOp().getKind() == Kind.EQUAL:
                    negated_assertions.append(assertion.eqTerm(assertion.getChild(0), assertion.getChild(1)).notTerm())
                elif assertion.getOp().getKind() == Kind.GEQ:
                    negated_assertions.append(assertion.geqTerm(assertion.getChild(0), assertion.getChild(1)).notTerm())
                elif assertion.getOp().getKind() == Kind.GT:
                    negated_assertions.append(assertion.gtTerm(assertion.getChild(0), assertion.getChild(1)).notTerm())
                elif assertion.getOp().getKind() == Kind.LEQ:
                    negated_assertions.append(assertion.leqTerm(assertion.getChild(0), assertion.getChild(1)).notTerm())
                elif assertion.getOp().getKind() == Kind.LT:
                    negated_assertions.append(assertion.ltTerm(assertion.getChild(0), assertion.getChild(1)).notTerm())
                else:
                    raise ValueError("Unsupported assertion type: {}".format(assertion))
        return negated_assertions
