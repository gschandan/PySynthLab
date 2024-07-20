import dataclasses
import logging
import typing
from datetime import datetime
from typing import List, Dict, Tuple, Set, Callable, Collection, Any

import pyparsing
from z3 import *

from src.helpers.parser.src.exceptions import ParseException
from src.helpers.parser.src.v1.parser import SygusV1Parser
from src.utilities.options import Options
from src.helpers.parser.src import ast
from src.helpers.parser.src.ast import Program, CommandKind
from src.helpers.parser.src.resolution import FunctionDescriptor, FunctionKind, SortDescriptor
from src.helpers.parser.src.symbol_table_builder import SymbolTableBuilder
from src.helpers.parser.src.v2.parser import SygusV2Parser
from src.helpers.parser.src.v2.printer import SygusV2ASTPrinter


@dataclasses.dataclass
class SynthesisProblemContext:
    enumerator_solver: Solver = dataclasses.field(default_factory=Solver)
    verification_solver: Solver = dataclasses.field(default_factory=Solver)
    original_assertions: List[ExprRef] = dataclasses.field(default_factory=list)
    constraints: List[ast.Command] = dataclasses.field(default_factory=list)
    z3_variables: Dict[str, ExprRef] = dataclasses.field(default_factory=dict)
    z3_synth_functions: Dict[str, FuncDeclRef] = dataclasses.field(default_factory=dict)
    z3_synth_function_args: Dict[str, Dict[str, ExprRef]] = dataclasses.field(default_factory=dict)
    z3_predefined_functions: Dict[str, Tuple[FuncDeclRef, ExprRef | FuncDeclRef | bool | int]] = dataclasses.field(
        default_factory=dict)
    z3_predefined_function_args: Dict[str, Dict[str, ExprRef]] = dataclasses.field(default_factory=dict)
    z3_constraints: List[ExprRef] = dataclasses.field(default_factory=list)
    assertions: Set[ExprRef] = dataclasses.field(default_factory=set)
    counterexamples: List[
        Tuple[QuantifierRef | ExprRef | Callable | Any, Dict[str, ExprRef], ExprRef]] = dataclasses.field(
        default_factory=list)
    z3_negated_constraints: Set[ExprRef] = dataclasses.field(default_factory=set)
    additional_constraints: List[ExprRef] = dataclasses.field(default_factory=list)
    synth_functions: List[Dict[str, typing.Union[str, List[ExprRef], SortRef]]] = dataclasses.field(
        default_factory=list)
    smt_problem: str = ""
    variable_mapping_dict: Dict[str, Dict[z3.ExprRef, z3.ExprRef]] = dataclasses.field(default_factory=dict)
    all_z3_functions: Dict[str, z3.FuncDeclRef] = dataclasses.field(default=dict)


class SynthesisProblem:
    """
    A class representing a synthesis problem in the SyGuS format.
    """
    pyparsing.ParserElement.enablePackrat()
    logger: logging.Logger = None
    options: Options = None

    def __init__(self, problem: str, options: Options = None):
        """
        Initialize a SynthesisProblem instance.

        :param problem: The input problem in the SyGuS format.
        :param options: Additional options (default: None).
        """
        if SynthesisProblem.logger is None:
            SynthesisProblem.setup_logger(options)

        self.input_problem: str = problem
        self.options = options or Options()
        self.parser, self.problem = self.parse_problem(problem)
        self.symbol_table = SymbolTableBuilder.run(self.problem)
        self.printer = SygusV2ASTPrinter(self.symbol_table)

        self.context = SynthesisProblemContext()
        self.context.enumerator_solver.set('smt.macro_finder', True)
        self.context.verification_solver.set('smt.macro_finder', True)
        if (self.options.synthesis_parameters.random_seed is not None and
                not self.options.synthesis_parameters.randomise_each_iteration):
            self.context.enumerator_solver.set('random_seed', self.options.synthesis_parameters.random_seed)
            self.context.verification_solver.set('random_seed', self.options.synthesis_parameters.random_seed)

        self.context.smt_problem = self.convert_sygus_to_smt()
        self.context.constraints = [x for x in self.problem.commands if x.command_kind == CommandKind.CONSTRAINT]

        self.initialise_z3_variables()
        self.initialise_z3_synth_functions()
        self.initialise_z3_predefined_functions()
        self.map_z3_variables()
        self.populate_all_z3_functions()
        self.parse_constraints()

    @staticmethod
    def setup_logger(options: Options = None):
        """
        Setup the logger for the SynthesisProblem class.
        :param options: The options object containing logging configurations.
        """
        if SynthesisProblem.logger is not None:
            return

        logger = logging.getLogger(__name__)
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        if options and options.logging.file:
            log_file = options.logging.file
        else:
            log_file = os.path.join(log_dir, f"run_{datetime.now()}.log")

        file_handler = logging.FileHandler(log_file)
        console_handler = logging.StreamHandler()

        log_level = options.logging.level if options else logging.DEBUG
        file_handler.setLevel(log_level)
        console_handler.setLevel(log_level)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(log_level)

        SynthesisProblem.logger = logger
        SynthesisProblem.options = options

    @staticmethod
    def parse_problem(problem: str) -> tuple[SygusV1Parser | SygusV2Parser, Program]:
        """
        Attempt to parse the problem string using SygusV1Parser and SygusV2Parser.

        :param problem: The input problem string.
        :return: A tuple containing the successful parser and the parsed problem.
        :raises ParseException: If both parsers fail to parse the problem.
        """
        try:
            parser = SygusV1Parser()
            parsed_problem = parser.parse(problem)
            SynthesisProblem.logger.info("Successfully parsed with SygusV1Parser")
            return parser, parsed_problem
        except ParseException as e:
            SynthesisProblem.logger.warning(f"Failed to parse with SygusV1Parser: {e}")
            SynthesisProblem.logger.warning("Attempting to parse with SygusV2Parser")

            try:
                parser = SygusV2Parser()
                parsed_problem = parser.parse(problem)
                SynthesisProblem.logger.info("Successfully parsed with SygusV2Parser")
                return parser, parsed_problem
            except ParseException as e:
                SynthesisProblem.logger.error(f"Failed to parse with both SygusV1Parser and SygusV2Parser: {e}")
                raise ParseException(f"Failed to parse with both SygusV1Parser and SygusV2Parser: {e}")

    def __str__(self, as_smt=False) -> str:
        """
        Return the string representation of the synthesis problem.

        :param as_smt: If True, return the problem in SMT-LIB format.
        :return: The string representation of the problem.
        """
        if as_smt:
            return self.context.smt_problem
        return self.printer.run(self.problem, self.symbol_table)

    def info_sygus(self) -> None:
        """
        Print the string representation of the synthesis problem.
        """
        SynthesisProblem.logger.info(str(self))

    def info_smt(self) -> None:
        """
        Print the string representation of the synthesis problem in SMT-LIB format.
        """
        SynthesisProblem.logger.info(self.__str__(as_smt=True))

    def convert_sygus_to_smt(self) -> str:
        """
        Convert the synthesis problem from SyGuS format to SMT-LIB format.

        This method parses the input SyGuS problem, transforms it into SMT-LIB format,
        and returns the result as a string.

        :return: The synthesis problem in SMT-LIB format.

        Example:
            parser = SynthesisProblemParser(sygus_problem)
            smt_problem = parser.convert_sygus_to_smt()
            print(smt_problem)

            # Output:
            # (set-logic LIA)
            # (declare-fun max2 (Int Int) Int)
            # (declare-var x Int)
            # (declare-var y Int)
            # (assert (and (>= (max2 x y) x) (>= (max2 x y) y) (or (= x (max2 x y)) (= y (max2 x y)))))
            # (check-sat)
        """
        i_expr = pyparsing.QuotedString(quoteChar='"') | pyparsing.QuotedString(quoteChar='|', unquoteResults=False)
        s_expr = pyparsing.nestedExpr(opener='(', closer=')', ignoreExpr=i_expr)
        s_expr.ignore(';' + pyparsing.restOfLine)

        sygus_parser = pyparsing.ZeroOrMore(s_expr)
        sygus_ast = sygus_parser.parseString(self.input_problem, parseAll=True).asList()

        constraints = []
        constraint_indices = []
        for i, statement in enumerate(sygus_ast):
            if statement[0] == 'constraint':
                constraints.append(statement[1])
                constraint_indices.append(i)
            elif statement[0] == 'check-synth':
                statement[0] = 'check-sat'
            elif statement[0] == 'synth-fun':
                statement[0] = 'declare-fun'
                statement[2] = [var_decl[1] for var_decl in statement[2]]
        if constraints:
            conjoined_constraints = ['and'] + constraints
            sygus_ast[constraint_indices[0]] = ['assert', conjoined_constraints]
            for index in reversed(constraint_indices[1:]):
                del sygus_ast[index]

        def serialise(line):
            return line if type(line) is not list else f'({" ".join(serialise(expression) for expression in line)})'

        return '\n'.join(serialise(statement) for statement in sygus_ast)

    def get_logic(self) -> str:
        """
        Get the logic of the synthesis problem.

        :return: The logic name.
        """
        return self.symbol_table.logic_name

    def get_synth_funcs(self) -> Dict[str, FunctionDescriptor]:
        """
        Get the synthesis functions of the problem.

        :return: A dictionary mapping function names to their declaration commands.
        """
        return self.symbol_table.synth_functions

    def get_predefined_funcs(self) -> Dict[str, FunctionDescriptor]:
        """
        Get the predefined functions of the problem.

        :return: A dictionary mapping function names to their declaration commands.
        """
        return self.symbol_table.user_defined_functions

    def get_synth_func(self, symbol: str) -> FunctionDescriptor:
        """
        Get a specific synthesis function by its symbol.

        :param symbol: The symbol of the synthesis function.
        :return: The function declaration command.
        """
        return next(filter(lambda x:
                           x.function_kind == FunctionKind.SYNTH_FUN and x.identifier.symbol == symbol,
                           list(self.symbol_table.synth_functions.values())))

    def get_var_symbols(self) -> List[str]:
        """
        Get the variable symbols of the problem.

        :return: A list of variable symbols.
        """
        return [x.symbol for x in self.problem.commands if x.command_kind == CommandKind.DECLARE_VAR]

    def get_function_symbols(self) -> List[str]:
        """
        Get the function symbols of the problem.

        :return: A list of function symbols.
        """
        return [x.function_name for x in self.problem.commands if x.command_kind == CommandKind.DEFINE_FUN]

    def initialise_z3_variables(self) -> None:
        """
        Initialize Z3 variables.
        """
        for command in self.problem.commands:
            if command.command_kind == CommandKind.DECLARE_VAR and command.sort_expression.identifier.symbol == 'Int':
                self.context.z3_variables[command.symbol] = z3.Int(command.symbol)
            if command.command_kind == CommandKind.DECLARE_VAR and command.sort_expression.identifier.symbol == 'Bool':
                self.context.z3_variables[command.symbol] = z3.Bool(command.symbol)

    def initialise_z3_synth_functions(self) -> None:
        """Initialize Z3 synthesis functions."""
        for func in self.get_synth_funcs().values():
            z3_arg_sorts = [self.convert_sort_descriptor_to_z3_sort(s) for s in func.argument_sorts]
            z3_range_sort = self.convert_sort_descriptor_to_z3_sort(func.range_sort)

            args = [self.create_z3_variable(name, sort) for name, sort in zip(func.argument_names, z3_arg_sorts)]

            arg_mapping = dict(zip(func.argument_names, args))
            self.context.z3_synth_function_args[func.identifier.symbol] = arg_mapping
            self.context.z3_synth_functions[func.identifier.symbol] = z3.Function(
                func.identifier.symbol, *z3_arg_sorts, z3_range_sort
            )

    def initialise_z3_predefined_functions(self) -> None:
        """Initialize Z3 predefined functions."""
        for func in self.get_predefined_funcs().values():
            z3_arg_sorts = [self.convert_sort_descriptor_to_z3_sort(s) for s in func.argument_sorts]
            z3_range_sort = self.convert_sort_descriptor_to_z3_sort(func.range_sort)

            args = [self.create_z3_variable(name, sort) for name, sort in zip(func.argument_names, z3_arg_sorts)]
            local_variables = dict(zip(func.argument_names, args))

            parsed_body = self.parse_term(func.function_body, local_variables)  # Use local_variables here

            self.context.z3_predefined_function_args[func.identifier.symbol] = local_variables

            self.context.z3_predefined_functions[func.identifier.symbol] = (
                z3.Function(func.identifier.symbol, *z3_arg_sorts, z3_range_sort),
                parsed_body,
            )

    def populate_all_z3_functions(self) -> None:
        """Add all parsed Z3 functions to a dictionary."""
        self.context.all_z3_functions = dict(self.context.z3_synth_functions.items())
        predefined_function_dict = {name: func_tuple[0] for name, func_tuple in
                                    self.context.z3_predefined_functions.items()}
        self.context.all_z3_functions.update(predefined_function_dict)

    def map_z3_variables(self) -> None:
        """
        Map z3 variables.
        """
        for func_name, func in self.context.z3_synth_functions.items():
            free_variables = [z3.Var(i, func.domain(i)) for i in range(func.arity())]
            declared_variables = list(self.context.z3_variables.values())
            variable_mapping = {free_var: declared_var for free_var, declared_var in
                                zip(free_variables, declared_variables)}
            self.context.variable_mapping_dict[func_name] = variable_mapping

    @staticmethod
    def negate_assertions(assertions: List[z3.ExprRef]) -> List[z3.ExprRef]:
        """
        Negate a list of assertions.

        :param assertions: The list of assertions to negate.
        :return: The negated assertions.
        """
        negated_assertions = []
        for assertion in assertions:
            args = assertion.num_args()
            if z3.is_and(assertion) or z3.is_or(assertion) or z3.is_not(assertion):
                if args > 1:
                    negated_children = [z3.Not(assertion.arg(i)) for i in range(args)]
                    negated_assertions.append(z3.Or(*negated_children))
                else:
                    negated_assertions.append(z3.Not(assertion))
            elif z3.is_expr(assertion) and args == 2:
                if z3.is_eq(assertion):
                    negated_assertions.append(assertion.arg(0) != assertion.arg(1))
                elif z3.is_ge(assertion):
                    negated_assertions.append(assertion.arg(0) < assertion.arg(1))
                elif z3.is_gt(assertion):
                    negated_assertions.append(assertion.arg(0) <= assertion.arg(1))
                elif z3.is_le(assertion):
                    negated_assertions.append(assertion.arg(0) > assertion.arg(1))
                elif z3.is_lt(assertion):
                    negated_assertions.append(assertion.arg(0) >= assertion.arg(1))
                else:
                    raise ValueError("Unsupported assertion type: {}".format(assertion))
        return negated_assertions

    def parse_constraints(self) -> None:
        """
        Parse the constraint_substitution of the synthesis problem.
        """
        all_constraints = []
        declared_variables = set(self.get_var_symbols())
        declared_functions = set(self.get_function_symbols())
        declared_synth_functions = set(self.get_synth_funcs().keys())

        for constraint in self.context.constraints:
            if isinstance(constraint, ast.ConstraintCommand):
                undeclared_variables = self.find_undeclared_variables(constraint.constraint, declared_variables,
                                                                      declared_functions, declared_synth_functions)
                if undeclared_variables:
                    raise ValueError(f"Undeclared variables used in constraint: {', '.join(undeclared_variables)}")
                term = self.parse_term(constraint.constraint)
                all_constraints.append(term)
                self.context.original_assertions.append(term)

        if all_constraints:
            combined_constraint = z3.And(*all_constraints)
            self.context.z3_constraints.append(combined_constraint)
        else:
            SynthesisProblem.logger.info("Warning: No constraints found or generated.")

        self.context.z3_negated_constraints = self.negate_assertions(self.context.z3_constraints)

    def find_undeclared_variables(self, term, declared_variables, declared_functions, declared_synth_functions):
        """
        Find undeclared variables in a term.

        :param term: The term to check.
        :param declared_variables: The set of declared variables.
        :param declared_functions: The set of declared functions.
        :param declared_synth_functions: The set of declared synthesis functions.
        :return: A list of undeclared variables found in the term.
        """
        undeclared_variables = []

        if isinstance(term, ast.IdentifierTerm):
            symbol = term.identifier.symbol
            if (symbol not in declared_variables and symbol not in declared_functions
                    and symbol not in declared_synth_functions):
                undeclared_variables.append(symbol)
        elif isinstance(term, ast.FunctionApplicationTerm):
            for arg in term.arguments:
                undeclared_variables.extend(self.find_undeclared_variables(arg, declared_variables, declared_functions,
                                                                           declared_synth_functions))
        elif isinstance(term, ast.QuantifiedTerm):
            for var_name, _ in term.quantified_variables:
                if var_name not in declared_variables:
                    undeclared_variables.append(var_name)
            undeclared_variables.extend(
                self.find_undeclared_variables(term.term_body, declared_variables, declared_functions,
                                               declared_synth_functions))

        return undeclared_variables

    def parse_term(self, term: ast.Term,
                   local_variables: Dict[str, ExprRef] = None) -> ExprRef | FuncDeclRef | bool | int:
        """
        Parse a term with optional local variable context.

        :param term: The term to parse.
        :param local_variables: A dictionary of local variables to consider during parsing.
        :return: The Z3 expression representing the term.
        """
        local_variables = local_variables if local_variables else {}

        if isinstance(term, ast.IdentifierTerm):
            symbol = term.identifier.symbol
            if symbol in local_variables:
                return local_variables[symbol]
            elif symbol in self.context.z3_variables:
                return self.context.z3_variables[symbol]
            elif symbol in self.context.z3_predefined_function_args:
                return self.context.z3_variables[symbol]
            elif symbol in self.context.z3_synth_functions:
                return self.context.z3_synth_functions[symbol]
            elif symbol in self.context.z3_predefined_functions:
                return self.context.z3_predefined_functions[symbol][0]
            else:
                raise ValueError(f"Undefined symbol: {symbol}")
        elif isinstance(term, ast.LiteralTerm):
            literal = term.literal
            if literal.literal_kind == ast.LiteralKind.NUMERAL:
                return z3.IntVal(int(literal.literal_value))
            elif literal.literal_kind == ast.LiteralKind.BOOLEAN:
                return z3.BoolVal(literal.literal_value.lower() == "true")
            else:
                raise ValueError(f"Unsupported literal kind: {literal.literal_kind}")
        elif isinstance(term, ast.FunctionApplicationTerm):
            func_symbol = term.function_identifier.symbol
            nested_local_variables = local_variables.copy()
            if func_symbol in self.context.z3_predefined_function_args:
                for arg_name, z3_var in self.context.z3_predefined_function_args[func_symbol].items():
                    nested_local_variables[arg_name] = z3_var

            args = [self.parse_term(arg, nested_local_variables) for arg in term.arguments]

            operator_map = {
                "and": lambda *_args: z3.And(*_args),
                "or": lambda *_args: z3.Or(*_args),
                "not": lambda arg: z3.Not(arg),
                ">": lambda arg1, arg2: arg1 > arg2,
                "<": lambda arg1, arg2: arg1 < arg2,
                ">=": lambda arg1, arg2: arg1 >= arg2,
                "<=": lambda arg1, arg2: arg1 <= arg2,
                "+": lambda *_args: z3.Sum(_args),
                "*": lambda *_args: z3.Product(*_args),
                "/": lambda arg1, arg2: arg1 / arg2,
                "=>": lambda arg1, arg2: z3.Implies(arg1, arg2),
                "ite": lambda cond, arg1, arg2: z3.If(cond, arg1, arg2),
            }
            if func_symbol in operator_map:
                op = operator_map[func_symbol]
                if func_symbol == "not":
                    assert len(args) == 1, "'not' should have 1 argument"
                    return op(args[0])
                else:
                    assert len(args) >= 2, f"'{func_symbol}' should have at least 2 arguments"
                    return op(*args)
            elif func_symbol == "-":
                if len(args) == 1:
                    return -args[0]
                elif len(args) == 2:
                    return args[0] - args[1]
                raise ValueError("Minus operator '-' should have 1 or 2 arguments")
            elif func_symbol == "=":
                if len(args) == 2:
                    return args[0] == args[1]
                else:
                    return z3.And(*[args[i] == args[i + 1] for i in range(len(args) - 1)])
            elif func_symbol in self.context.z3_synth_functions:
                func_term = self.context.z3_synth_functions[func_symbol]
                return func_term(*args)
            elif func_symbol in self.context.z3_predefined_functions:
                func, body = self.context.z3_predefined_functions[func_symbol]
                function_args = self.context.z3_predefined_function_args[func_symbol]
                substituted_body = z3.substitute(body,
                                                 [(arg, value) for arg, value in zip(function_args.values(), args)])
                return substituted_body
            else:
                raise ValueError(f"Undefined function symbol: {func_symbol}")
        elif isinstance(term, ast.QuantifiedTerm):
            quantified_variables = []
            for var_name, _ in term.quantified_variables:
                if var_name in self.context.z3_variables:
                    quantified_variables.append(self.context.z3_variables[var_name])
                else:
                    raise ValueError(f"Undeclared variable used in quantifier: {var_name}")
            body = self.parse_term(term.term_body)
            if term.quantifier_kind == ast.QuantifierKind.FORALL:
                return z3.ForAll(quantified_variables, body)
            elif term.quantifier_kind == ast.QuantifierKind.EXISTS:
                return z3.Exists(quantified_variables, body)
            else:
                raise ValueError(f"Unsupported quantifier kind: {term.quantifier_kind}")
        else:
            raise ValueError(f"Unsupported term type: {type(term)}")

    @staticmethod
    def convert_sort_descriptor_to_z3_sort(sort_descriptor: SortDescriptor) -> z3.SortRef | None:
        """
        Convert a sort descriptor to a Z3 sort.

        :param sort_descriptor: The sort descriptor to convert.
        :return: The corresponding Z3 sort.
        """
        sort_symbol = sort_descriptor.identifier.symbol
        return {
            'Int': z3.IntSort(),
            'Bool': z3.BoolSort(),
        }.get(sort_symbol, None)

    @staticmethod
    def create_z3_variable(name: str, sort: z3.SortRef) -> z3.ExprRef:
        """
        Create a Z3 variable of the given sort.

        :param name: The name of the variable.
        :param sort: The sort of the variable.
        :return: The Z3 expression representing the variable.
        """
        if sort == z3.IntSort():
            return z3.Int(name)
        elif sort == z3.BoolSort():
            return z3.Bool(name)
        else:
            raise ValueError(f"Unsupported sort: {sort}")

    def collect_function_io_pairs(self, func: z3.FuncDeclRef) -> List[Tuple[Dict[str, z3.ExprRef], z3.ExprRef]]:
        """
        Collect input-output pairs for a function from the constraints.

        :param func: The function to collect input-output pairs for.
        :return: A list of input-output pairs.
        """
        io_pairs = []
        for constraint in self.context.constraints:
            if isinstance(constraint, ast.ConstraintCommand) and isinstance(constraint.constraint,
                                                                            ast.FunctionApplicationTerm):
                if constraint.constraint.function_identifier.symbol == func.name():
                    example_inputs = {arg.identifier.symbol: self.parse_term(arg) for arg in
                                      constraint.constraint.arguments[:-1]}
                    example_output = self.parse_term(constraint.constraint.arguments[-1])
                    io_pairs.append((example_inputs, example_output))
        return io_pairs

    def substitute_constraints(self, constraints: Collection[z3.ExprRef],
                               functions_to_replace: List[z3.FuncDeclRef],
                               replacement_expressions: List[
                                   typing.Union[z3.FuncDeclRef, z3.ExprRef, Callable]]) -> \
            List[z3.ExprRef]:
        """
        Substitute candidate expressions into a list of constraints.

        :param constraints: The list of constraints to substitute.
        :param functions_to_replace: The list of functions to replace.
        :param replacement_expressions: The replacement expressions.
        :return: The list of substituted constraints.
        """
        synth_substitutions = list(zip(functions_to_replace, replacement_expressions))
        predefined_substitutions = [(func, body) for func, body in self.context.z3_predefined_functions.values()]

        substituted_constraints = []
        for constraint in constraints:
            substituted_constraint = z3.substitute_funs(constraint, [*synth_substitutions])
            substituted_constraint = z3.substitute_funs(substituted_constraint, [*predefined_substitutions])
            substituted_constraints.append(substituted_constraint)

        return substituted_constraints

    def substitute_candidates(self, constraints: Collection[z3.ExprRef],
                              candidate_functions: List[tuple[z3.FuncDeclRef,
                              typing.Union[z3.FuncDeclRef, z3.ExprRef, Callable]]]) -> List[z3.ExprRef]:
        """
        Substitute candidate expressions into a list of constraints.

        :param constraints: The list of constraints to substitute.
        :param candidate_functions: The candidate functions and their replacements.
        :return: The list of substituted constraints.
        """
        predefined_substitutions = [(func, body) for func, body in self.context.z3_predefined_functions.values()]

        substituted_constraints = []
        for constraint in constraints:
            synth_substituted = z3.substitute_funs(constraint, candidate_functions)
            predefined_substituted = z3.substitute_funs(synth_substituted, predefined_substitutions)
            substituted_constraints.append(predefined_substituted)
        return substituted_constraints
