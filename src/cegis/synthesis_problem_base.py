import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Union, Any, List
import pyparsing

from src.helpers.parser.src import symbol_table_builder
from src.helpers.parser.src.ast import Program, CommandKind, Term
from src.helpers.parser.src.resolution import FunctionDescriptor
from src.helpers.parser.src.v1.parser import SygusV1Parser
from src.helpers.parser.src.v2.parser import SygusV2Parser
from src.helpers.parser.src.v2.printer import SygusV2ASTPrinter
from src.utilities.options import Options, LoggingOptions


class BaseSynthesisProblem(ABC):
    pyparsing.ParserElement.enablePackrat()
    logger: logging.Logger = None
    options: Options = None

    def __init__(self, problem: str, options: Options = None):
        self.options = options or Options()
        if self.logger is None:
            self.setup_logger(self.options)

        self.parser, self.problem = self.parse_problem(problem)
        self.symbol_table = symbol_table_builder.SymbolTableBuilder.run(self.problem)
        self.printer = SygusV2ASTPrinter(self.symbol_table)

        self.input_problem = problem
        self.smt_problem = self.convert_sygus_to_smt()
        self.constraints = [x for x in self.problem.commands if x.command_kind == CommandKind.CONSTRAINT]

    @classmethod
    def setup_logger(cls, options: Options = None):
        """
        Setup the logger for the SynthesisProblem class.

        This method configures the logging system for the SynthesisProblem class,
        setting up both file and console handlers with appropriate log levels and formats.

        Args:
            options (Options, optional): The options object containing logging configurations. Defaults to None.

        Example:
            >>> problem_str = "(set-logic LIA)\\n(synth-fun max2 ((x Int) (y Int)) Int)\\n(declare-var a Int)\\n(declare-var b Int)\\n(constraint (>= (max2 a b) a))"
            >>> options = Options(logging=LoggingOptions(file='synthesis.log', level="DEBUG"))
            >>> problem = BaseSynthesisProblem(problem_str, options)
            >>> problem.setup_logger(options)
        """
        if cls.logger is not None:
            return

        cls.logger = logging.getLogger(__name__)
        log_level = options.logging.level if options else logging.DEBUG
        cls.logger.setLevel(log_level)

        project_root = Path(__file__).resolve().parent.parent.parent
        log_dir = project_root / "logs"
        log_dir.mkdir(exist_ok=True)

        if options and options.logging.file:
            log_file = options.logging.file
        else:
            log_file = log_dir / f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        file_handler = logging.FileHandler(log_file)
        console_handler = logging.StreamHandler()

        file_handler.setLevel(log_level)
        console_handler.setLevel(log_level)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        cls.logger.addHandler(file_handler)
        cls.logger.addHandler(console_handler)

    @staticmethod
    def parse_problem(problem: str) -> Tuple[Union[SygusV1Parser, SygusV2Parser], Program]:
        """
        Attempt to parse the problem string using SygusV1Parser and SygusV2Parser.

        This method tries to parse the input problem using SygusV1Parser first,
        and if that fails, it attempts to parse with SygusV2Parser.

        Args:
            problem (str): The input problem string in SyGuS format.

        Returns:
            tuple: A tuple containing the successful parser and the parsed problem.

        Raises:
            ParseException: If both parsers fail to parse the problem.

        Example:
            >>> problem_str = "(set-logic LIA)\\n(synth-fun max2 ((x Int) (y Int)) Int ...)"
            >>> parser, parsed_problem = BaseSynthesisProblem.parse_problem(problem_str)
            >>> print(type(parser))
            <class 'src.helpers.parser.src.v1.parser.SygusV1Parser'>
            >>> print(type(parsed_problem))
            <class 'src.helpers.parser.src.ast.Program'>
        """
        try:
            parser = SygusV1Parser()
            parsed_problem = parser.parse(problem)
            return parser, parsed_problem
        except Exception:
            try:
                parser = SygusV2Parser()
                parsed_problem = parser.parse(problem)
                return parser, parsed_problem
            except Exception as e:
                raise ValueError(f"Failed to parse problem: {e}")

    def convert_sygus_to_smt(self) -> str:
        """
        Convert the synthesis problem from SyGuS format to SMT-LIB format.

        This method parses the input SyGuS problem, transforms it into SMT-LIB format,
        and returns the result as a string.

        Returns:
            str: The synthesis problem in SMT-LIB format.

        Example:
            >>> problem_str = "(set-logic LIA)\\n(synth-fun max2 ((x Int) (y Int)) Int)\\n(declare-var a Int)\\n(declare-var b Int)\\n(constraint (>= (max2 a b) a))\\n(check-synth)"
            >>> synthesis_problem = BaseSynthesisProblem(problem_str)
            >>> print(synthesis_problem.convert_sygus_to_smt())
            (set-logic LIA)
            (declare-fun max2 (Int Int) Int)
            (declare-var a Int)
            (declare-var b Int)
            (assert (>= (max2 a b) a))
            (check-sat)
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
            if len(constraints) > 1:
                sygus_ast[constraint_indices[0]] = ['assert', ['and'] + constraints]
                for index in reversed(constraint_indices[1:]):
                    del sygus_ast[index]
            else:
                sygus_ast[constraint_indices[0]] = ['assert', constraints[0]]

        def serialise(line):
            return line if isinstance(line, str) else f'({" ".join(serialise(expression) for expression in line)})'

        return '\n'.join(serialise(statement) for statement in sygus_ast)

    def __str__(self, as_smt=False) -> str:
        """
        Return the string representation of the synthesis problem.

        Args:
            as_smt (bool, optional): If True, return the problem in SMT-LIB format. Defaults to False.

        Returns:
            str: The string representation of the problem.

        Example:
            >>> problem_str = "(set-logic LIA)\\n(synth-fun max2 ((x Int) (y Int)) Int ...)"
            >>> synthesis_problem = BaseSynthesisProblem(problem_str)
            >>> print(synthesis_problem)
            (set-logic LIA)
            (synth-fun max2 ((x Int) (y Int)) Int ...)
            >>> print(synthesis_problem.__str__(as_smt=True))
            (set-logic LIA)
            (declare-fun max2 (Int Int) Int)
            ...
        """
        if as_smt:
            return self.smt_problem
        return self.printer.run(self.problem, self.symbol_table)

    def info_sygus(self) -> None:
        """
        Log the string representation of the synthesis problem in SyGuS format.

        This method logs the SyGuS representation of the synthesis problem.

        Example:
            >>> problem_str = "(set-logic LIA)\\n(synth-fun max2 ((x Int) (y Int)) Int ...)"
            >>> synthesis_problem = BaseSynthesisProblem(problem_str)
            >>> synthesis_problem.info_sygus()
            # Output in log:
            # (set-logic LIA)
            # (synth-fun max2 ((x Int) (y Int)) Int ...)
        """
        self.logger.info(str(self))

    def info_smt(self) -> None:
        """
        Log the string representation of the synthesis problem in SMT-LIB format.

        This method logs the SMT-LIB representation of the synthesis problem.

        Example:
            >>> problem_str = "(set-logic LIA)\\n(synth-fun max2 ((x Int) (y Int)) Int ...)"
            >>> synthesis_problem = BaseSynthesisProblem(problem_str)
            >>> synthesis_problem.info_smt()
            # Output in log:
            # (set-logic LIA)
            # (declare-fun max2 (Int Int) Int)
            # ...
        """
        self.logger.info(self.__str__(as_smt=True))

    def get_logic(self) -> str:
        """
        Get the logic of the synthesis problem.

        Returns:
            str: The logic name.

        Example:
            >>> problem_str = "(set-logic LIA)\\n(synth-fun max2 ((x Int) (y Int)) Int ...)"
            >>> synthesis_problem = BaseSynthesisProblem(problem_str)
            >>> print(synthesis_problem.get_logic())
            LIA
        """
        return self.symbol_table.logic_name

    def get_synth_funcs(self) -> dict[str, FunctionDescriptor]:
        """
        Get the synthesis functions of the problem.

        Returns:
            Dict[str, FunctionDescriptor]: A dictionary mapping function names to their declaration commands.

        Example:
            >>> problem_str = "(set-logic LIA)\\n(synth-fun max2 ((x Int) (y Int)) Int ...)"
            >>> synthesis_problem = BaseSynthesisProblem(problem_str)
            >>> synth_funcs = synthesis_problem.get_synth_funcs()
            >>> print(list(synth_funcs.keys()))
            ['max2']
            >>> print(synth_funcs['max2'].argument_sorts)
            [SortDescriptor(identifier=Identifier(symbol='Int')), SortDescriptor(identifier=Identifier(symbol='Int'))]
        """
        return self.symbol_table.synth_functions

    def get_predefined_funcs(self) -> dict[str, FunctionDescriptor]:
        """
        Get the predefined functions of the problem.

        Returns:
            Dict[str, FunctionDescriptor]: A dictionary mapping function names to their declaration commands.

        Example:
            >>> problem_str = "(set-logic LIA)\\n(define-fun min2 ((x Int) (y Int)) Int (ite (<= x y) x y))"
            >>> synthesis_problem = BaseSynthesisProblem(problem_str)
            >>> predefined_funcs = synthesis_problem.get_predefined_funcs()
            >>> print(list(predefined_funcs.keys()))
            ['min2']
            >>> print(predefined_funcs['min2'].function_body)
            FunctionApplicationTerm(function_identifier=Identifier(symbol='ite'), arguments=[...])
        """
        return self.symbol_table.user_defined_functions

    def get_var_symbols(self) -> list[str]:
        """
        Get the variable names decalred by the problem.

        Returns:
            List[str]: A list of variable symbols.

        Example:
            >>> problem_str = "(set-logic LIA)\\n(declare-var x Int)\\n(declare-var y Int)"
            >>> synthesis_problem = BaseSynthesisProblem(problem_str)
            >>> print(synthesis_problem.get_var_symbols())
            ['x', 'y']
        """
        return [x.symbol for x in self.problem.commands if x.command_kind == CommandKind.DECLARE_VAR]

    def get_function_symbols(self) -> list[str]:
        """
        Get the function symbols declared by the problem.

        Returns:
            List[str]: A list of function symbols.

        Example:
            >>> problem_str = "(set-logic LIA)\\n(define-fun f ((x Int)) Int (+ x 1))"
            >>> synthesis_problem = BaseSynthesisProblem(problem_str)
            >>> print(synthesis_problem.get_function_symbols())
            ['f']
        """
        return [x.function_name for x in self.problem.commands if x.command_kind == CommandKind.DEFINE_FUN]

    @abstractmethod
    def initialise_variables(self):
        pass

    @abstractmethod
    def initialise_synth_functions(self):
        pass

    @abstractmethod
    def initialise_predefined_functions(self):
        pass

    @abstractmethod
    def parse_constraints(self):
        pass

    @abstractmethod
    def parse_term(self, term: Term, local_variables: Dict[str, Any] = None) -> Any:
        pass

    @abstractmethod
    def substitute_constraints(self, constraints, functions_to_replace, replacement_expressions):
        pass

