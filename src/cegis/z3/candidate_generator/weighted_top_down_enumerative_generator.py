from itertools import product
from typing import List, Tuple, Union
import z3


class TopDownCandidateGenerator:
    def __init__(self, problem: 'SynthesisProblem'):
        self.grammar = None
        self.problem = problem
        self.min_const = problem.options.synthesis_parameters.min_const
        self.max_const = problem.options.synthesis_parameters.max_const
        self.max_depth = problem.options.synthesis_parameters.max_depth
        self.explored_expressions: dict[str, set[str]] = {func_name: set() for func_name in
                                                          problem.context.variable_mapping_dict.keys()}

    def define_grammar(self, variables):
        return {
            'S': [
                ('T', 1),
                (('ite', 'B', 'S', 'S'), 10),
                (('+', 'S', 'S'), 5),
                (('-', 'S', 'S'), 5),
                (('*', 'S', 'S'), 6),
                (('Neg', 'S'), 3)
            ],
            'B': [
                (('>', 'T', 'T'), 2),
                (('>=', 'T', 'T'), 2),
                (('<', 'T', 'T'), 2),
                (('<=', 'T', 'T'), 2),
                (('==', 'T', 'T'), 2),
                (('!=', 'T', 'T'), 2)
            ],
            'T': [(var, 1) for var in variables] + [(str(i), 1) for i in range(self.min_const, self.max_const + 1)]
        }

    def generate_candidates(self) -> List[Tuple[z3.ExprRef, str]]:
        candidates = []
        for func_name, variable_mapping in self.problem.context.variable_mapping_dict.items():
            variables = [str(var) for var in variable_mapping.values()]
            grammar = self.define_grammar(variables)
            self.problem.logger.debug(f'Grammar {grammar}')
            expansions = self.expand(grammar, ('S', 0), 0)
            sorted_expansions = sorted(expansions, key=lambda x: x[1], reverse=True)
            for candidate, complexity in sorted_expansions:
                simplified_expr = self.simplify_term(candidate)
                expr_str = str(simplified_expr)
                if expr_str not in self.explored_expressions[func_name]:
                    self.problem.logger.debug(f'Generated candidate: {expr_str} (complexity: {complexity})')
                    self.explored_expressions[func_name].add(expr_str)
                    candidates.append((simplified_expr, func_name))
                    break

        return candidates

    def expand(self, grammar, expr, depth, complexity=0):
        if depth > self.max_depth:
            return []

        expansions = []

        if isinstance(expr, tuple):
            if len(expr) == 2 and isinstance(expr[1], int):
                symbol, expr_complexity = expr
            else:
                symbol, expr_complexity = expr, 0
        else:
            symbol, expr_complexity = expr, 0

        if symbol in grammar:
            sorted_productions = sorted(grammar[symbol], key=lambda x: x[1], reverse=True)
            for production, prod_complexity in sorted_productions:
                new_complexity = complexity + prod_complexity
                sub_expansions = self.expand(grammar, production, depth + 1, new_complexity)
                expansions.extend(sub_expansions)
        elif isinstance(symbol, str):
            return [(z3.Int(symbol), complexity + expr_complexity)]
        elif isinstance(symbol, tuple):
            op, *args = symbol
            arg_expansions = [self.expand(grammar, arg, depth + 1, complexity) for arg in args]

            for arg_combo in product(*arg_expansions):
                args = [arg[0] for arg in arg_combo]
                new_complexity = sum(arg[1] for arg in arg_combo) + expr_complexity
                if op == '+':
                    expansions.append((args[0] + args[1], new_complexity))
                elif op == '-':
                    expansions.append((args[0] - args[1], new_complexity))
                elif op == '*':
                    if isinstance(args[0], z3.IntNumRef) or isinstance(args[1], z3.IntNumRef):
                        expansions.append((args[0] * args[1], new_complexity))
                elif op == 'ite':
                    expansions.append((z3.If(args[0], args[1], args[2]), new_complexity))
                elif op == '>':
                    expansions.append((args[0] > args[1], new_complexity))
                elif op == '>=':
                    expansions.append((args[0] >= args[1], new_complexity))
                elif op == '<=':
                    expansions.append((args[0] <= args[1], new_complexity))
                elif op == '<':
                    expansions.append((args[0] < args[1], new_complexity))
                elif op == '==':
                    expansions.append((args[0] == args[1], new_complexity))
                elif op == '!=':
                    expansions.append((args[0] != args[1], new_complexity))
                elif op == 'Neg':
                    expansions.append((-args[0], new_complexity))

        return expansions

    def simplify_term(self, term: Union[z3.ExprRef, int]) -> Union[z3.ExprRef, int]:
        if isinstance(term, z3.ExprRef):
            return z3.simplify(term)
        return term

    def prune_candidates(self, candidates: List[Tuple[z3.ExprRef, str]]) -> List[Tuple[z3.ExprRef, str]]:
        return candidates
