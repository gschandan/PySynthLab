from typing import List, Tuple, Union, Optional
import z3

from src.cegis.z3.candidate_generator.candidate_generator_base import CandidateGenerator
from src.cegis.z3.synthesis_problem_z3 import SynthesisProblemZ3


# experimental
class PartialTopDownCandidateGenerator(CandidateGenerator):
    def __init__(self, problem: SynthesisProblemZ3):
        super().__init__(problem)
        self.problem = problem
        self.grammar = problem.options.synthesis_parameters.custom_grammar
        self.min_const = problem.options.synthesis_parameters.min_const
        self.max_const = problem.options.synthesis_parameters.max_const
        self.max_depth = problem.options.synthesis_parameters.max_depth
        self.explored_expressions: dict[str, dict[str, float]] = {func_name: {} for func_name in
                                                                  problem.context.variable_mapping_dict.keys()}
        self.metrics = problem.metrics
        self.logger = self.problem.logger
        self.memo: dict[tuple[str, int], z3.ExprRef | None] = {}

    def define_grammar(self, variables: list[str]) -> dict[str, list]:
        if self.grammar is None:
            return {
                'S': [('T', 10), (('ite', 'B', 'S', 'S'), 50), (('+', 'S', 'S'), 30),
                      (('-', 'S', 'S'), 30), (('*', 'S', 'S'), 35), (('neg', 'S'), 20)],
                'B': [(('>', 'T', 'T'), 15), (('>=', 'T', 'T'), 15), (('<', 'T', 'T'), 15),
                      (('<=', 'T', 'T'), 15), (('==', 'T', 'T'), 15), (('!=', 'T', 'T'), 15)],
                'T': [(var, 5) for var in variables] + [(str(i), 5) for i in range(self.min_const, self.max_const + 1)]
            }
        return self.grammar

    def generate_candidates(self) -> List[Tuple[z3.ExprRef, str]]:
        candidates = []
        for func_name, variable_mapping in self.problem.context.variable_mapping_dict.items():
            variables = [str(var) for var in variable_mapping.values()]
            grammar = self.define_grammar(variables)
            self.logger.info(f"\nTop-down search process for {func_name} with grammar {grammar}:")
            for depth in range(1, self.max_depth + 1):
                self.logger.info(f"Searching at depth {depth}")
                candidate = self.top_down_search(grammar, 'S', depth, func_name, variable_mapping)
                if candidate is not None:
                    self.logger.info(f"Potentially viable candidate found: {candidate}")
                    candidates.append((candidate, func_name))
                    break
            if not candidates:
                self.logger.info("No candidate found")
        return candidates

    def top_down_search(self, grammar: dict, symbol: str, depth: int, func_name: str, variable_mapping: dict) -> Optional[z3.ExprRef]:
        if depth == 0:
            return None

        if (symbol, depth) in self.memo:
            return self.memo[(symbol, depth)]

        self.logger.info(f"Exploring: {symbol} (Depth: {depth})")

        candidates = []

        if symbol in grammar:
            expansions = grammar[symbol]
        elif symbol in variable_mapping:
            return variable_mapping[symbol]
        elif symbol.isdigit() or (symbol[0] == '-' and symbol[1:].isdigit()):
            return z3.IntVal(int(symbol))
        else:
            self.logger.debug(f"No valid expansion found for {symbol} at depth {depth}")
            return None

        for expansion, weight in sorted(expansions, key=lambda x: x[1], reverse=True):
            self.logger.debug(f"Trying expansion: {expansion} (Weight: {weight})")
            if isinstance(expansion, tuple):
                op, *args = expansion
                arg_results = []
                for arg in args:
                    arg_result = self.top_down_search(grammar, arg, depth - 1, func_name, variable_mapping)
                    if arg_result is None:
                        break
                    arg_results.append(arg_result)

                if len(arg_results) == len(args):
                    result = self.apply_operator(op, arg_results)
                    if result is not None:
                        candidates.append(result)
            else:
                result = self.top_down_search(grammar, expansion, depth - 1, func_name, variable_mapping)
                if result is not None:
                    candidates.append(result)

        best_candidate = None
        best_score = float('-inf')

        for candidate in candidates:
            score = self.evaluate_candidate(candidate, func_name)
            self.logger.info(f"  Candidate: {self.format_expr(candidate)}, Score: {score}")
            if score > best_score:
                best_score = score
                best_candidate = candidate

        if best_score >= 0.5:
            self.memo[(symbol, depth)] = best_candidate
            return best_candidate

        self.logger.debug(f"No valid expansion found for {symbol} at depth {depth}")
        self.memo[(symbol, depth)] = None
        return None

    def apply_operator(self, op: str, args: List[z3.ExprRef]) -> Optional[z3.ExprRef]:
        try:
            if op == '+':
                return args[0] + args[1]
            elif op == '-':
                return args[0] - args[1]
            elif op == '*':
                return args[0] * args[1]
            elif op == 'neg':
                return -args[0]
            elif op == 'ite':
                return z3.If(args[0], args[1], args[2])
            elif op in ['>', '>=', '<', '<=', '==', '!=']:
                return getattr(args[0], op)(args[1])
            else:
                self.logger.warning(f"Unknown operator: {op}")
                return None
        except Exception as e:
            self.logger.warning(f"Error applying operator {op}: {e}")
            return None

    def format_expr(self, expr: Union[z3.ExprRef, str, tuple]) -> str:
        if isinstance(expr, (z3.ArithRef, z3.BoolRef)):
            return str(expr)
        elif isinstance(expr, str):
            return expr
        elif isinstance(expr, tuple):
            return f"{expr[0]}({', '.join(self.format_expr(arg) for arg in expr[1:])})"
        else:
            return "?"

    def evaluate_candidate(self, candidate: z3.ExprRef, func_name: str) -> float:
        score = 0.5 * self.check_partial_satisfaction(candidate, func_name) + \
                0.5 * self.fuzzy_satisfaction(candidate, func_name)
        return score

    @staticmethod
    def simplify_term(term: Union[z3.ExprRef, int]) -> Union[z3.ExprRef, int]:
        if isinstance(term, z3.ExprRef):
            return z3.simplify(term)
        return term

    def prune_candidates(self, candidates: List[Tuple[z3.ExprRef, str]]) -> List[Tuple[z3.ExprRef, str]]:
        return candidates
