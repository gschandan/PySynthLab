import cvc5
from cvc5 import Kind, Term

from src.cegis.synthesis_problem_base import BaseSynthesisProblem
from src.helpers.parser.src import ast
from src.helpers.parser.src.ast import QuantifiedTerm, FunctionApplicationTerm, LiteralTerm, IdentifierTerm, CommandKind
from src.helpers.parser.src.resolution import SortDescriptor


class SynthesisProblemCVC5(BaseSynthesisProblem):

    def __init__(self, problem: str, options: object = None):
        super().__init__(problem, options)

        self.solver = cvc5.Solver()
        self.solver.setLogic("ALL")
        self.solver.setOption("produce-models", "true")
        self.solver.setOption("sygus", "true")

        self.cvc5_variables = {}
        self.cvc5_synth_functions = {}
        self.cvc5_predefined_functions = {}
        self.cvc5_constraints = []

        self.initialise_variables()
        self.initialise_synth_functions()
        self.initialise_predefined_functions()
        self.parse_constraints()

    def initialise_variables(self):
        for command in self.problem.commands:
            if command.command_kind == CommandKind.DECLARE_VAR:
                sort = self.convert_sort_descriptor_to_cvc5_sort(command.sort_expression)
                self.cvc5_variables[command.symbol] = self.solver.mkConst(sort, command.symbol)

    def initialise_synth_functions(self):
        for func in self.symbol_table.synth_functions.values():
            arg_sorts = [self.convert_sort_descriptor_to_cvc5_sort(s) for s in func.argument_sorts]
            range_sort = self.convert_sort_descriptor_to_cvc5_sort(func.range_sort)
            func_sort = self.solver.mkFunctionSort(arg_sorts, range_sort)
            self.cvc5_synth_functions[func.identifier.symbol] = self.solver.mkConst(func_sort, func.identifier.symbol)

    def initialise_predefined_functions(self):
        for func in self.symbol_table.user_defined_functions.values():
            arg_sorts = [self.convert_sort_descriptor_to_cvc5_sort(s) for s in func.argument_sorts]
            range_sort = self.convert_sort_descriptor_to_cvc5_sort(func.range_sort)
            func_sort = self.solver.mkFunctionSort(arg_sorts, range_sort)
            self.cvc5_predefined_functions[func.identifier.symbol] = (
                self.solver.mkConst(func_sort, func.identifier.symbol),
                self.parse_term(func.function_body)
            )

    def parse_constraints(self):
        for constraint in self.constraints:
            if isinstance(constraint, ast.ConstraintCommand):
                term = self.parse_term(constraint.constraint)
                self.cvc5_constraints.append(term)

    def parse_term(self, term: Term, local_variables: dict[str, cvc5.Term] = None) -> cvc5.Term:
        local_variables = local_variables or {}
        if isinstance(term, IdentifierTerm):
            symbol = term.identifier.symbol
            if symbol in local_variables:
                return local_variables[symbol]
            elif symbol in self.cvc5_variables:
                return self.cvc5_variables[symbol]
            elif symbol in self.cvc5_predefined_functions:
                return self.cvc5_predefined_functions[symbol][0]
            elif symbol in self.cvc5_synth_functions:
                return self.cvc5_synth_functions[symbol]
            else:
                raise ValueError(f"Undefined symbol: {symbol}")
        elif isinstance(term, LiteralTerm):
            literal = term.literal
            if literal.literal_kind == ast.LiteralKind.NUMERAL:
                return self.solver.mkInteger(int(literal.literal_value))
            elif literal.literal_kind == ast.LiteralKind.BOOLEAN:
                return self.solver.mkBoolean(literal.literal_value.lower() == "true")
            else:
                raise ValueError(f"Unsupported literal kind: {literal.literal_kind}")
        elif isinstance(term, FunctionApplicationTerm):
            func_symbol = term.function_identifier.symbol
            args = [self.parse_term(arg, local_variables) for arg in term.arguments]

            operator_map = {
                "and": Kind.AND,
                "or": Kind.OR,
                "not": Kind.NOT,
                "=": Kind.EQUAL,
                ">": Kind.GT,
                "<": Kind.LT,
                ">=": Kind.GEQ,
                "<=": Kind.LEQ,
                "+": Kind.ADD,
                "*": Kind.MULT,
                "/": Kind.INTS_DIVISION,
                "-": Kind.SUB,
                "ite": Kind.ITE,
            }

            if func_symbol in operator_map:
                return self.solver.mkTerm(operator_map[func_symbol], *args)
            elif func_symbol in self.cvc5_synth_functions:
                func_term = self.cvc5_synth_functions[func_symbol]
                return self.solver.mkTerm(Kind.APPLY_UF, func_term, *args)
            elif func_symbol in self.cvc5_predefined_functions:
                func, body = self.cvc5_predefined_functions[func_symbol]
                return self.solver.mkTerm(Kind.APPLY_UF, func, *args)
            else:
                raise ValueError(f"Undefined function symbol: {func_symbol}")
        elif isinstance(term, QuantifiedTerm):
            bound_vars = [self.solver.mkVar(self.convert_sort_descriptor_to_cvc5_sort(var[1]), var[0]) for var in term.quantified_variables]
            body = self.parse_term(term.term_body, {**local_variables, **{var.getSymbol(): var for var in bound_vars}})
            if term.quantifier_kind == ast.QuantifierKind.FORALL:
                return self.solver.mkTerm(Kind.FORALL, self.solver.mkTerm(Kind.BOUND_VAR_LIST, *bound_vars), body)
            elif term.quantifier_kind == ast.QuantifierKind.EXISTS:
                return self.solver.mkTerm(Kind.EXISTS, self.solver.mkTerm(Kind.BOUND_VAR_LIST, *bound_vars), body)
            else:
                raise ValueError(f"Unsupported quantifier kind: {term.quantifier_kind}")
        else:
            raise ValueError(f"Unsupported term type: {type(term)}")

    def convert_sort_descriptor_to_cvc5_sort(self, sort_descriptor: SortDescriptor) -> cvc5.Sort:
        sort_symbol = sort_descriptor.identifier.symbol
        return {
            'Int': self.solver.getIntegerSort(),
            'Bool': self.solver.getBooleanSort(),
        }.get(sort_symbol, self.solver.getNullSort())

    def substitute_constraints(self, constraints, functions_to_replace, replacement_expressions):
        substituted_constraints = []
        for constraint in constraints:
            substituted_constraint = constraint
            for func, replacement in zip(functions_to_replace, replacement_expressions):
                substituted_constraint = substituted_constraint.substitute(func, replacement)
            substituted_constraints.append(substituted_constraint)
        return substituted_constraints
