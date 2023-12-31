import z3
import pyparsing
from pysynthlab.helpers.parser.src import symbol_table_builder
from pysynthlab.helpers.parser.src.ast import Program, CommandKind, FunctionApplicationTerm, LiteralTerm, \
    Identifier, LiteralKind
from pysynthlab.helpers.parser.src.resolution import SymbolTable, FunctionKind
from pysynthlab.helpers.parser.src.v1.parser import SygusV1Parser
from pysynthlab.helpers.parser.src.v1.printer import SygusV1ASTPrinter
from pysynthlab.helpers.parser.src.v2.parser import SygusV2Parser
from pysynthlab.helpers.parser.src.v2.printer import SygusV2ASTPrinter


class SynthesisProblem:
    """
        A class representing a synthesis problem.
        ...
        Attributes
        ----------
        options : object
            Additional options for the synthesis problem.
        sygus_standard : int
            The selected SyGuS-IF standard version.
        parser : SygusParser (SygusV1Parser or SygusV2Parser)
            The SyGuS parser based on the chosen standard.
        problem : Program
            The parsed synthesis problem.
        symbol_table : SymbolTable
            The symbol table built from the parsed problem.
        printer : SygusASTPrinter (SygusV1ASTPrinter or SygusV2ASTPrinter)
            The AST (Abstract Syntax Tree) printer based on the chosen standard.
        ...
        Methods
        -------
        info():
            Prints the synthesis problem to the console
        """

    def __init__(self, problem: str, solver: z3.Solver, sygus_standard: int = 1, options: object = None):
        """
        Initialize a SynthesisProblem instance with the provided parameters.

        Parameters
        ----------
        problem : str
            The synthesis problem to be parsed.
        sygus_standard : int, optional
            The SyGuS-IF standard version (1 or 2). Default is 2.
        options : object, optional
            Additional options for version 1 (e.g., 'no-unary-minus'). Default is None.

        Examples
        --------
        >> problem = SynthesisProblem("(set-logic LIA)\n(synth-fun f ((x Int) (y Int)) Int)\n...", sygus_standard=1)
        >> print(problem.problem)
        (set-logic LIA)
        (synth-fun f ((x Int) (y Int)) Int)
        ...

        """
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
        self.z3variables = []
        self.z3functions = []
        self.z3function_definitions = []
        self.constraints = [x for x in self.problem.commands if x.command_kind == CommandKind.CONSTRAINT]
        pyparsing.ParserElement.enablePackrat()
        self.smt_problem = self.convert_sygus_to_smt()

    def __str__(self) -> str:
        """
        Returns a string representation of the synthesis problem.
        :returns: str: A string representation of the synthesis problem.
        """
        return self.printer.run(self.problem, self.symbol_table)

    def info(self) -> None:
        """ Prints the synthesis problem to the console
        :returns: None
        """
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
        for variable in [x for x in self.problem.commands if x.command_kind == CommandKind.DECLARE_VAR]:
            if variable.__getattribute__('sort_expression').identifier.symbol == 'Int':
                self.z3variables.append(z3.Int(variable.symbol, self.solver.ctx))

    def initialise_functions(self):
        for function in [x for x in self.problem.commands if
                         x.command_kind == CommandKind.DECLARE_FUN or x.command_kind == CommandKind.DEFINE_FUN]:
            function_name = function.function_name
            function_return_sort = sort_map[function.__getattribute__('function_range_sort').identifier.symbol]
            function_params = [sort_map[sort[1].identifier.symbol] for sort in
                               function.function_parameters]
            function_body = translate_ast_to_z3(function.function_body, function_name)
            self.z3functions.append(z3.Function(function_name, *function_params, function_return_sort))
        print(self.z3functions[0])

    def setup_solver(self):
        self.initialise_variables()
        self.initialise_functions()
        for function in [x for x in self.problem.commands if
                         x.command_kind == CommandKind.DECLARE_FUN or x.command_kind == CommandKind.DEFINE_FUN]:
            function.__str__()


def translate_ast_to_z3(node, function_name):
    if isinstance(node, LiteralTerm):
        return map_literal_to_z3_val(node.literal.literal_kind, node.literal.literal_value)
    elif isinstance(node, Identifier):
        return z3_operator_map[node.symbol]
    elif isinstance(node, FunctionApplicationTerm):
        args = [translate_ast_to_z3(arg, function_name) for arg in node.arguments]


z3_operator_map = {
    '+': z3.Z3_OP_BADD,
    '-': z3.Z3_OP_BSUB,
    '*': z3.Z3_OP_BMUL,
    '/': z3.Z3_OP_BSDIV,
    '%': z3.Z3_OP_BSMOD,
    'True': z3.Z3_OP_TRUE,
    'False': z3.Z3_OP_FALSE,
    '==': z3.Z3_OP_EQ,
    'Distinct': z3.Z3_OP_DISTINCT,
    'If': z3.Z3_OP_ITE,
    'And': z3.Z3_OP_AND,
    'Or': z3.Z3_OP_OR,
    'Xor': z3.Z3_OP_XOR,
    'Not': z3.Z3_OP_NOT,
    'Implies': z3.Z3_OP_IMPLIES,
    'ToReal': z3.Z3_OP_TO_REAL,
    'ToInt': z3.Z3_OP_TO_INT,
    '**': z3.Z3_OP_POWER,
    'IsInt': z3.Z3_OP_IS_INT,
    '|': z3.Z3_OP_BOR,
    '&': z3.Z3_OP_BAND,
    '~': z3.Z3_OP_BNOT,
    '^': z3.Z3_OP_BXOR,
    '<=': z3.Z3_OP_SLEQ,
    '<': z3.Z3_OP_SLT,
    '>=': z3.Z3_OP_SGEQ,
    '>': z3.Z3_OP_SGT,
    'ULE': z3.Z3_OP_ULEQ,
    'ULT': z3.Z3_OP_ULT,
    'UGE': z3.Z3_OP_UGEQ,
    'UGT': z3.Z3_OP_UGT,
}

sort_map = {
    'Int': z3.IntSort(),
    'Bool': z3.BoolSort()
}


def map_literal_to_z3_val(literal_kind: LiteralKind, literal_value: object) -> object:
    if literal_kind == LiteralKind.NUMERAL:
        return z3.IntVal(literal_value)
    elif literal_kind == LiteralKind.BOOLEAN:
        return z3.BoolVal(literal_value)
    else:
        raise ValueError(f"Unsupported kind: {literal_kind}")
