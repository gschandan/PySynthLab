import dataclasses
import logging
import typing
from typing import List, Dict, Tuple, Set, Callable, Collection, Any

from z3 import *

from src.cegis.synthesis_problem_base import BaseSynthesisProblem
from src.helpers.parser.src import ast
from src.helpers.parser.src.ast import Program, CommandKind
from src.helpers.parser.src.resolution import SortDescriptor
from src.helpers.parser.src.v1.parser import SygusV1Parser
from src.helpers.parser.src.v2.parser import SygusV2Parser
from src.helpers.parser.src.v2.printer import SygusV2ASTPrinter
from src.utilities.cancellation_token import GlobalCancellationToken
from src.utilities.options import Options


@dataclasses.dataclass
class SynthesisProblemZ3Context:
    """
    A dataclass to store the context of a synthesis problem.

    This class holds various solvers, constraints, variables, and other data
    structures used throughout the synthesis process.

    Attributes:
        enumerator_solver (Solver): Z3 solver for enumeration.
        verification_solver (Solver): Z3 solver for verification.
        original_assertions (List[ExprRef]): List of original Z3 assertions.
        constraints (List[ast.Command]): List of constraint commands.
        z3_variables (Dict[str, ExprRef]): Dictionary mapping variable names to Z3 expressions.
        z3_synth_functions (Dict[str, FuncDeclRef]): Dictionary mapping synthesis function names to Z3 function declarations.
        z3_synth_function_args (Dict[str, Dict[str, ExprRef]]): Dictionary mapping synthesis function names to their arguments.
        z3_predefined_functions (Dict[str, Tuple[FuncDeclRef, ExprRef | FuncDeclRef | bool | int]]): Dictionary mapping predefined function names to their Z3 representations.
        z3_predefined_function_args (Dict[str, Dict[str, ExprRef]]): Dictionary mapping predefined function names to their arguments.
        z3_constraints (List[ExprRef]): List of Z3 constraints conjoined with And.
        z3_non_conjoined_constraints (List[ExprRef]): List of Z3 constraints, non-conjoined.
        assertions (Set[ExprRef]): Set of Z3 assertions.
        counterexamples (List[Tuple[QuantifierRef | ExprRef | Callable | Any, Dict[str, ExprRef], ExprRef]]): List of counterexamples.
        z3_negated_constraints (Set[ExprRef]): Set of negated Z3 constraints.
        additional_constraints (List[ExprRef]): List of additional Z3 constraints.
        smt_problem (str): SMT-LIB representation of the problem.
        variable_mapping_dict (Dict[str, Dict[z3.ExprRef, z3.ExprRef]]): Dictionary mapping variable names to their Z3 representations.
        all_z3_functions (Dict[str, z3.FuncDeclRef]): Dictionary of all Z3 functions in the problem.
    """

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
    z3_non_conjoined_constraints: List[ExprRef] = dataclasses.field(default_factory=list)
    assertions: Set[ExprRef] = dataclasses.field(default_factory=set)
    counterexamples: List[
        Tuple[QuantifierRef | ExprRef | Callable | Any, Dict[str, ExprRef], ExprRef]] = dataclasses.field(
        default_factory=list)
    z3_negated_constraints: Set[ExprRef] = dataclasses.field(default_factory=set)
    additional_constraints: List[ExprRef] = dataclasses.field(default_factory=list)
    smt_problem: str = ""
    variable_mapping_dict: Dict[str, Dict[z3.ExprRef, z3.ExprRef]] = dataclasses.field(default_factory=dict)
    all_z3_functions: Dict[str, z3.FuncDeclRef] = dataclasses.field(default=dict)


class SynthesisProblemZ3(BaseSynthesisProblem):
    """
    A class representing a synthesis problem in the SyGuS format.

    This class provides methods for parsing, analyzing, and manipulating
    synthesis problems expressed in the SyGuS (Syntax-Guided Synthesis) format.
    It includes functionality for converting SyGuS to SMT-LIB format,
    initializing Z3 variables and functions, parsing constraints, substituting constraints.

    Attributes:
        logger (logging.Logger): Logger for the class.
        options (Options): Configuration options for the synthesis problem.
        input_problem (str): The original input problem in SyGuS format.
        parser (SygusV1Parser | SygusV2Parser): The parser used to parse the problem.
        problem (Program): The parsed SyGuS program.
        symbol_table (SymbolTable): Symbol table for the parsed program.
        printer (SygusV2ASTPrinter): Printer for the AST.
        context (SynthesisProblemZ3Context): Context holding various data structures and solvers.

    Example:
        .. code-block:: python
            >>> problem_str = "(set-logic LIA)\\n(synth-fun max2 ((x Int) (y Int)) Int ...)"
            >>> options = Options()
            >>> synthesis_problem = SynthesisProblemZ3(problem_str, options)
            >>> print(synthesis_problem.get_logic())
            LIA
            >>> print(synthesis_problem.get_synth_funcs())
            {'max2': FunctionDescriptor(...)}
    """

    def __init__(self, problem: str, options: Options = None):
        """
        Initialize a SynthesisProblem instance.

        This method sets up the synthesis problem by parsing the input,
        initializing the symbol table, and setting up various Z3 components.

        Args:
            problem (str): The input problem in the SyGuS format.
            options (Options, optional): Additional options for problem setup. Defaults to None.

        Raises:
            ParseException: If the problem cannot be parsed using either SygusV1Parser or SygusV2Parser.

        Example:
            >>> problem_str = "(set-logic LIA)\\n(synth-fun max2 ((x Int) (y Int)) Int ...)"
            >>> options = Options()
            >>> synthesis_problem = SynthesisProblemZ3(problem_str, options)
        """
        super().__init__(problem, options)

        self.context = SynthesisProblemZ3Context()
        self.context.enumerator_solver.set('timeout', self.options.solver.timeout)
        self.context.verification_solver.set('timeout', self.options.solver.timeout)
        self.context.enumerator_solver.set('smt.macro_finder', True)
        self.context.verification_solver.set('smt.macro_finder', True)
        if not self.options.synthesis_parameters.randomise_each_iteration:
            self.context.enumerator_solver.set('random_seed', self.options.synthesis_parameters.random_seed)
            self.context.verification_solver.set('random_seed', self.options.synthesis_parameters.random_seed)

        self.context.smt_problem = self.smt_problem
        self.context.constraints = self.constraints

        self.initialise_variables()
        self.initialise_synth_functions()
        self.initialise_predefined_functions()
        self.map_z3_variables()
        self.populate_all_z3_functions()
        self.parse_constraints()

    def initialise_variables(self) -> None:
        """
        Initialise Z3 variables based on the declared variables in the problem.

        This method creates Z3 variables for each declared variable in the problem
        and stores them in the context.

        Example:
            >>> problem_str = "(set-logic LIA)\\n(declare-var x Int)\\n(declare-var y Bool)"
            >>> synthesis_problem = SynthesisProblemZ3(problem_str)
            >>> synthesis_problem.initialise_variables()
            >>> print(synthesis_problem.context.z3_variables)
            {'x': Int('x'), 'y': Bool('y')}
        """
        for command in self.problem.commands:
            if command.command_kind == CommandKind.DECLARE_VAR and command.sort_expression.identifier.symbol == 'Int':
                self.context.z3_variables[command.symbol] = z3.Int(command.symbol)
            if command.command_kind == CommandKind.DECLARE_VAR and command.sort_expression.identifier.symbol == 'Bool':
                self.context.z3_variables[command.symbol] = z3.Bool(command.symbol)

    def initialise_synth_functions(self) -> None:
        """
        Initialise Z3 synthesis functions based on the synth-fun commands in the problem.

        This method creates Z3 function declarations for each synthesis function
        in the problem and stores them in the context.

        Example:
            >>> problem_str = "(set-logic LIA)\\n(synth-fun max2 ((x Int) (y Int)) Int)"
            >>> synthesis_problem = SynthesisProblemZ3(problem_str)
            >>> synthesis_problem.initialise_synth_functions()
            >>> print(synthesis_problem.context.z3_synth_functions)
            {'max2': Function('max2', IntSort(), IntSort(), IntSort())}
        """
        for func in self.get_synth_funcs().values():
            z3_arg_sorts = [self.convert_sort_descriptor_to_z3_sort(s) for s in func.argument_sorts]
            z3_range_sort = self.convert_sort_descriptor_to_z3_sort(func.range_sort)

            args = [self.create_z3_variable(name, sort) for name, sort in zip(func.argument_names, z3_arg_sorts)]

            arg_mapping = dict(zip(func.argument_names, args))
            self.context.z3_synth_function_args[func.identifier.symbol] = arg_mapping
            self.context.z3_synth_functions[func.identifier.symbol] = z3.Function(
                func.identifier.symbol, *z3_arg_sorts, z3_range_sort
            )

    def initialise_predefined_functions(self) -> None:
        """
        Initialize Z3 predefined functions based on the define-fun commands in the problem.

        This method creates Z3 function declarations for each predefined function
        in the problem and stores them in the context.

        Example:
            >>> problem_str = "(set-logic LIA)\\n(define-fun min2 ((x Int) (y Int)) Int (ite (<= x y) x y))"
            >>> synthesis_problem = SynthesisProblemZ3(problem_str)
            >>> synthesis_problem.initialise_z3_predefined_functions()
            >>> print(synthesis_problem.context.z3_predefined_functions)
            {'min2': (Function('min2', IntSort(), IntSort(), IntSort()), If(x <= y, x, y))}
        """
        for func in self.get_predefined_funcs().values():
            z3_arg_sorts = [self.convert_sort_descriptor_to_z3_sort(s) for s in func.argument_sorts]
            z3_range_sort = self.convert_sort_descriptor_to_z3_sort(func.range_sort)
            if not z3_arg_sorts:
                parsed_body = self.parse_term(func.function_body)
                const = z3.Const(func.identifier.symbol, z3_range_sort)
                self.context.z3_predefined_functions[func.identifier.symbol] = (const, parsed_body)
            else:
                args = [self.create_z3_variable(name, sort) for name, sort in zip(func.argument_names, z3_arg_sorts)]
                local_variables = dict(zip(func.argument_names, args))

                parsed_body = self.parse_term(func.function_body, local_variables)  # Use local_variables here

                self.context.z3_predefined_function_args[func.identifier.symbol] = local_variables

                self.context.z3_predefined_functions[func.identifier.symbol] = (
                    z3.Function(func.identifier.symbol, *z3_arg_sorts, z3_range_sort),
                    parsed_body
                )

    def populate_all_z3_functions(self) -> None:
        """
        Add all parsed Z3 functions to a dictionary.

        This method combines all synthesis functions and predefined functions
        into a single dictionary for easy access.

        Example:
            >>> problem_str = "(set-logic LIA)\\n(synth-fun max2 ((x Int) (y Int)) Int)\\n(define-fun min2 ((x Int) (y Int)) Int (ite (<= x y) x y))"
            >>> synthesis_problem = SynthesisProblemZ3(problem_str)
            >>> synthesis_problem.initialise_synth_functions()
            >>> synthesis_problem.initialise_predefined_functions()
            >>> synthesis_problem.populate_all_z3_functions()
            >>> print(list(synthesis_problem.context.all_z3_functions.keys()))
            ['max2', 'min2']
        """
        self.context.all_z3_functions = dict(self.context.z3_synth_functions.items())
        predefined_function_dict = {name: func_tuple[0] for name, func_tuple in
                                    self.context.z3_predefined_functions.items()}
        self.context.all_z3_functions.update(predefined_function_dict)

    def map_z3_variables(self) -> None:
        """
        Map Z3 variables for each synthesis function.

        This method creates a mapping between free variables and declared variables
        for each synthesis function.

        Example:
            >>> problem_str = "(set-logic LIA)\\n(synth-fun max2 ((x Int) (y Int)) Int)\\n(declare-var a Int)\\n(declare-var b Int)"
            >>> synthesis_problem = SynthesisProblemZ3(problem_str)
            >>> synthesis_problem.initialise_variables()
            >>> synthesis_problem.initialise_synth_functions()
            >>> synthesis_problem.map_z3_variables()
            >>> print(synthesis_problem.context.variable_mapping_dict['max2'])
            {Var(0, IntSort()): Int('a'), Var(1, IntSort()): Int('b')}
        """
        for func_name, func in self.context.z3_synth_functions.items():
            free_variables = [z3.Var(i, func.domain(i)) for i in range(func.arity())]
            declared_variables = list(self.context.z3_variables.values())
            variable_mapping = {free_var: declared_var for free_var, declared_var in
                                zip(free_variables, declared_variables)}
            self.context.variable_mapping_dict[func_name] = variable_mapping

    def parse_constraints(self) -> None:
        """
        Parse the constraints of the synthesis problem.

        This method processes all constraint commands in the problem, checks for
        undeclared variables, and adds the parsed constraints to the context.

        Example:
            >>> problem_str = "(set-logic LIA)\\n(synth-fun max2 ((x Int) (y Int)) Int)\\n(declare-var a Int)\\n(declare-var b Int)\\n(constraint (>= (max2 a b) a))"
            >>> synthesis_problem = SynthesisProblemZ3(problem_str)
            >>> synthesis_problem.parse_constraints()
            >>> print(synthesis_problem.context.z3_constraints)
            [And(max2(a, b) >= a)]
        """

        all_constraints = []
        non_conjoined_constraints = []
        declared_variables = set(self.get_var_symbols())
        declared_functions = set(self.get_function_symbols())
        declared_synth_functions = set(self.get_synth_funcs().keys())

        for constraint in self.context.constraints:
            GlobalCancellationToken.check_cancellation()
            if isinstance(constraint, ast.ConstraintCommand):
                undeclared_variables = self.find_undeclared_variables(constraint.constraint, declared_variables,
                                                                      declared_functions, declared_synth_functions)
                if undeclared_variables:
                    self.logger.error(f"Undeclared variables used in constraint: {', '.join(undeclared_variables)}")
                    raise ValueError(f"Undeclared variables used in constraint: {', '.join(undeclared_variables)}")
                term = self.parse_term(constraint.constraint)
                all_constraints.append(term)
                self.context.original_assertions.append(term)

                if z3.is_and(term):
                    non_conjoined_constraints.extend(term.children())
                else:
                    non_conjoined_constraints.append(term)

        if all_constraints:
            self.context.z3_constraints = [z3.And(*all_constraints)]
        else:
            SynthesisProblemZ3.logger.info("Warning: No constraints found or generated.")

        self.context.z3_non_conjoined_constraints = non_conjoined_constraints
        self.context.z3_negated_constraints = self.negate_assertions(self.context.z3_constraints)

    def split_conjoined_constraint(self, constraint: z3.ExprRef) -> List[z3.ExprRef]:
        """
        Split a potentially conjoined constraint into its individual components.

        Args:
            constraint (z3.ExprRef): The constraint to split.

        Returns:
            List[z3.ExprRef]: A list of individual constraints.

        Example:
            >>> problem_str = "(set-logic LIA)\\n(synth-fun f ((x Int)) Int)\\n(constraint (and (>= (f 0) 0) (>= (f 1) 1)))"
            >>> synthesis_problem = SynthesisProblemZ3(problem_str)
            >>> conjoined_constraint = synthesis_problem.context.z3_constraints[0]
            >>> split_constraints = synthesis_problem.split_conjoined_constraint(conjoined_constraint)
            >>> print(split_constraints)
            [f(0) >= 0, f(1) >= 1]
        """
        if z3.is_and(constraint):
            return list(constraint.children())
        else:
            return [constraint]

    def find_undeclared_variables(self, term, declared_variables, declared_functions, declared_synth_functions):
        """
        Find undeclared variables in a term.

        Args:
            term: The term to check.
            declared_variables: The set of declared variables.
            declared_functions: The set of declared functions.
            declared_synth_functions: The set of declared synthesis functions.

        Returns:
            List[str]: A list of undeclared variables found in the term.

        Example:
            >>> problem_str = "(set-logic LIA)\\n(synth-fun max2 ((x Int) (y Int)) Int)\\n(declare-var a Int)\\n(constraint (>= (max2 a b) a))"
            >>> synthesis_problem = SynthesisProblemZ3(problem_str)
            >>> constraint = synthesis_problem.context.constraints[0]
            >>> undeclared = synthesis_problem.find_undeclared_variables(constraint.constraint, {'a'}, set(), {'max2'})
            >>> print(undeclared)
            ['b']
        """
        GlobalCancellationToken.check_cancellation()

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

        Args:
            term (ast.Term): The term to parse.
            local_variables (Dict[str, ExprRef], optional): A dictionary of local variables to consider during parsing.

        Returns:
            ExprRef | FuncDeclRef | bool | int: The Z3 expression representing the term.

        Raises:
            ValueError: If an undefined symbol or unsupported term type is encountered.

        Example:
            >>> problem_str = "(set-logic LIA)\\n(synth-fun max2 ((x Int) (y Int)) Int)\\n(declare-var a Int)\\n(constraint (>= (max2 a 5) a))"
            >>> synthesis_problem = SynthesisProblemZ3(problem_str)
            >>> constraint = synthesis_problem.context.constraints[0]
            >>> parsed_term = synthesis_problem.parse_term(constraint.constraint)
            >>> print(parsed_term)
            max2(a, 5) >= a
        """
        GlobalCancellationToken.check_cancellation()
        local_variables = local_variables if local_variables else {}
        if isinstance(term, ast.IdentifierTerm):
            symbol = term.identifier.symbol
            if symbol in local_variables:
                return local_variables[symbol]
            elif symbol in self.context.z3_variables:
                return self.context.z3_variables[symbol]
            elif symbol in self.context.z3_predefined_functions:
                func, _ = self.context.z3_predefined_functions[symbol]
                return func
            elif symbol in self.context.z3_synth_functions:
                return self.context.z3_synth_functions[symbol]
            else:
                self.logger.error(f"Undefined symbol: {symbol}")
                raise ValueError(f"Undefined symbol: {symbol}")
        elif isinstance(term, ast.LiteralTerm):
            literal = term.literal
            if literal.literal_kind == ast.LiteralKind.NUMERAL:
                return z3.IntVal(int(literal.literal_value))
            elif literal.literal_kind == ast.LiteralKind.BOOLEAN:
                return z3.BoolVal(literal.literal_value.lower() == "true")
            else:
                self.logger.error(f"Unsupported literal kind: {literal.literal_kind}")
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
                self.logger.error("Minus operator '-' should have 1 or 2 arguments")
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
                self.logger.error(f"Undefined function symbol: {func_symbol}")
                raise ValueError(f"Undefined function symbol: {func_symbol}")
        elif isinstance(term, ast.QuantifiedTerm):
            quantified_variables = []
            for var_name, _ in term.quantified_variables:
                if var_name in self.context.z3_variables:
                    quantified_variables.append(self.context.z3_variables[var_name])
                else:
                    self.logger.error(f"Undeclared variable used in quantifier: {var_name}")
                    raise ValueError(f"Undeclared variable used in quantifier: {var_name}")
            body = self.parse_term(term.term_body)
            if term.quantifier_kind == ast.QuantifierKind.FORALL:
                return z3.ForAll(quantified_variables, body)
            elif term.quantifier_kind == ast.QuantifierKind.EXISTS:
                return z3.Exists(quantified_variables, body)
            else:
                self.logger.error(f"Unsupported quantifier kind: {term.quantifier_kind}")
                raise ValueError(f"Unsupported quantifier kind: {term.quantifier_kind}")
        else:
            self.logger.error(f"Unsupported term type: {type(term)}")
            raise ValueError(f"Unsupported term type: {type(term)}")

    @staticmethod
    def negate_assertions(assertions: List[z3.ExprRef]) -> List[z3.ExprRef]:
        """
        Negate a list of assertions.

        Args:
            assertions (List[z3.ExprRef]): The list of assertions to negate.

        Returns:
            List[z3.ExprRef]: The negated assertions.

        Example:
            >>> from z3 import Int, And, Or
            >>> x, y = Int('x'), Int('y')
            >>> assertions = [And(x > 0, y > 0), Or(x < 10, y < 10)]
            >>> negated = SynthesisProblemZ3.negate_assertions(assertions)
            >>> print(negated)
            [Or(Not(x > 0), Not(y > 0)), And(Not(x < 10), Not(y < 10))]
        """
        GlobalCancellationToken.check_cancellation()
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
                    SynthesisProblemZ3.logger.error("Unsupported assertion type: {}".format(assertion))
                    raise ValueError("Unsupported assertion type: {}".format(assertion))
        return negated_assertions

    @staticmethod
    def convert_sort_descriptor_to_z3_sort(sort_descriptor: SortDescriptor) -> z3.SortRef | None:
        """
        Convert a sort descriptor to a Z3 sort.

        Args:
            sort_descriptor (SortDescriptor): The sort descriptor to convert.

        Returns:
            z3.SortRef | None: The corresponding Z3 sort, or None if not supported.

        Example:
            >>> from src.helpers.parser.src.resolution import SortDescriptor
            >>> from src.helpers.parser.src.ast import Identifier
            >>> int_sort = SortDescriptor(identifier=Identifier(symbol='Int'))
            >>> z3_sort = SynthesisProblemZ3.convert_sort_descriptor_to_z3_sort(int_sort)
            >>> print(z3_sort)
            Int
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

        Args:
            name (str): The name of the variable.
            sort (z3.SortRef): The sort of the variable.

        Returns:
            z3.ExprRef: The Z3 expression representing the variable.

        Raises:
            ValueError: If the sort is not supported.

        Example:
            >>> int_var = SynthesisProblemZ3.create_z3_variable('x', z3.IntSort())
            >>> print(int_var)
            x
            >>> bool_var = SynthesisProblemZ3.create_z3_variable('y', z3.BoolSort())
            >>> print(bool_var)
            y
        """
        if sort == z3.IntSort():
            return z3.Int(name)
        elif sort == z3.BoolSort():
            return z3.Bool(name)
        else:
            SynthesisProblemZ3.logger.error(f"Unsupported sort: {sort}")
            raise ValueError(f"Unsupported sort: {sort}")

    def collect_function_io_pairs(self, func: z3.FuncDeclRef) -> List[Tuple[Dict[str, z3.ExprRef], z3.ExprRef]]:
        """
        Collect input-output pairs for a function from the constraints.

        Args:
            func (z3.FuncDeclRef): The function to collect input-output pairs for.

        Returns:
            List[Tuple[Dict[str, z3.ExprRef], z3.ExprRef]]: A list of input-output pairs.

        Example:
            >>> problem_str = "(set-logic LIA)\\n(synth-fun max2 ((x Int) (y Int)) Int)\\n(constraint (>= (max2 1 2) 2))\\n(constraint (= (max2 3 4) 4))"
            >>> synthesis_problem = SynthesisProblemZ3(problem_str)
            >>> max2_func = synthesis_problem.context.z3_synth_functions['max2']
            >>> io_pairs = synthesis_problem.collect_function_io_pairs(max2_func)
            >>> print(io_pairs)
            [({'x': 1, 'y': 2}, 2), ({'x': 3, 'y': 4}, 4)]
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

        Args:
            constraints (Collection[z3.ExprRef]): The list of constraints to substitute.
            functions_to_replace (List[z3.FuncDeclRef]): The list of functions to replace.
            replacement_expressions (List[typing.Union[z3.FuncDeclRef, z3.ExprRef, Callable]]): The replacement expressions.

        Returns:
            List[z3.ExprRef]: The list of substituted constraints.

        Example:
            >>> problem_str = "(set-logic LIA)\\n(synth-fun max2 ((x Int) (y Int)) Int)\\n(constraint (>= (max2 a b) a))"
            >>> synthesis_problem = SynthesisProblemZ3(problem_str)
            >>> max2 = synthesis_problem.context.z3_synth_functions['max2']
            >>> a, b = Int('a'), Int('b')
            >>> replacement = lambda x, y: If(x > y, x, y)
            >>> substituted = synthesis_problem.substitute_constraints(synthesis_problem.context.z3_constraints, [max2], [replacement])
            >>> print(substituted[0])
            If(a > b, a, b) >= a
        """
        GlobalCancellationToken.check_cancellation()

        synth_substitutions = list(zip(functions_to_replace, replacement_expressions))

        substituted_constraints = []
        for constraint in constraints:
            substituted_constraint = z3.substitute_funs(constraint, [*synth_substitutions])
            substituted_constraints.append(substituted_constraint)

        return substituted_constraints

    def substitute_candidates(self, constraints: Collection[z3.ExprRef],
                              candidate_functions: List[tuple[z3.FuncDeclRef,
                              typing.Union[z3.FuncDeclRef, z3.ExprRef, Callable]]]) -> List[z3.ExprRef]:
        """
        Substitute candidate expressions into a list of constraints.

        This method substitutes both candidate functions and predefined functions
        into the given constraints.

        Args:
            constraints (Collection[z3.ExprRef]): The list of constraints to substitute.
            candidate_functions (List[tuple[z3.FuncDeclRef, typing.Union[z3.FuncDeclRef, z3.ExprRef, Callable]]]): 
                The candidate functions and their replacements.

        Returns:
            List[z3.ExprRef]: The list of substituted constraints.

        Example:
            >>> problem_str = "(set-logic LIA)\\n(synth-fun max2 ((x Int) (y Int)) Int)\\n(define-fun min2 ((x Int) (y Int)) Int (ite (<= x y) x y))\\n(constraint (>= (max2 (min2 a b) c) c))"
            >>> synthesis_problem = SynthesisProblemZ3(problem_str)
            >>> max2 = synthesis_problem.context.z3_synth_functions['max2']
            >>> a, b, c = Int('a'), Int('b'), Int('c')
            >>> replacement = lambda x, y: If(x > y, x, y)
            >>> candidates = [(max2, replacement)]
            >>> substituted = synthesis_problem.substitute_candidates(synthesis_problem.context.z3_constraints, candidates)
            >>> print(substituted[0])
            If(If(a <= b, a, b) > c, If(a <= b, a, b), c) >= c
        """
        GlobalCancellationToken.check_cancellation()
        predefined_substitutions = [(func, body) for func, body in self.context.z3_predefined_functions.values()]

        substituted_constraints = []
        for constraint in constraints:
            synth_substituted = z3.substitute_funs(constraint, candidate_functions)
            predefined_substituted = z3.substitute_funs(synth_substituted, predefined_substitutions)
            substituted_constraints.append(predefined_substituted)
        return substituted_constraints

    def partial_evaluate_constraint(self, constraint: z3.ExprRef, partial_assignment: Dict[z3.ExprRef, z3.ExprRef]) -> z3.ExprRef:
        """
        Partially evaluate a constraint given a partial assignment.

        Args:
            constraint (z3.ExprRef): The constraint to evaluate.
            partial_assignment (Dict[z3.ExprRef, z3.ExprRef]): A dictionary mapping variables to their partial assignments.

        Returns:
            z3.ExprRef: The partially evaluated constraint.

        Example:
            >>> problem_str = "(set-logic LIA)\\n(synth-fun f ((x Int)) Int)\\n(constraint (and (>= (f 0) 0) (>= (f 1) 1)))"
            >>> synthesis_problem = SynthesisProblemZ3(problem_str)
            >>> f = synthesis_problem.context.z3_synth_functions['f']
            >>> constraint = synthesis_problem.context.z3_non_conjoined_constraints[0]
            >>> partial_assignment = {f(0): z3.IntVal(5)}
            >>> evaluated = synthesis_problem.partial_evaluate_constraint(constraint, partial_assignment)
            >>> print(evaluated)
            5 >= 0
        """
        GlobalCancellationToken.check_cancellation()
        return z3.substitute(constraint, [(var, val) for var, val in partial_assignment.items()])
