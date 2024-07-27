from itertools import product
from typing import List, Tuple, Union
import z3


class Node:
    def __init__(self, value, weight=0):
        self.value = value
        self.weight = weight
        self.children = []


class WeightedTopDownCandidateGenerator:
    def __init__(self, problem: 'SynthesisProblem'):
        self.grammar = None
        self.problem = problem
        self.min_const = problem.options.synthesis_parameters.min_const
        self.max_const = problem.options.synthesis_parameters.max_const
        self.max_depth = problem.options.synthesis_parameters.max_depth
        self.explored_expressions: dict[str, set[str]] = {func_name: set() for func_name in
                                                          problem.context.variable_mapping_dict.keys()}

    def define_grammar(self, variables):
        # TODO: expose this as additional configuration
        return {
            'S': [
                ('T', 10),
                (('ite', 'B', 'S', 'S'), 50),
                (('+', 'S', 'S'), 30),
                (('-', 'S', 'S'), 30),
                (('*', 'S', 'S'), 35),
                (('Neg', 'S'), 20),
            ],
            'B': [
                (('>', 'T', 'T'), 15),
                (('>=', 'T', 'T'), 15),
                (('<', 'T', 'T'), 15),
                (('<=', 'T', 'T'), 15),
                (('==', 'T', 'T'), 15),
                (('!=', 'T', 'T'), 15),
            ],
            'T': [(var, 5) for var in variables] + [(str(i), 5) for i in range(self.min_const, self.max_const + 1)]
        }

    def generate_candidates(self) -> List[Tuple[z3.ExprRef, str]]:
        candidates = []
        for func_name, variable_mapping in self.problem.context.variable_mapping_dict.items():
            variables = [str(var) for var in variable_mapping.values()]
            grammar = self.define_grammar(variables)
            self.problem.logger.debug(f'Grammar {grammar}')

            tree = self.build_tree(grammar, 'S')
            tree_str = "\n".join(self.print_tree(tree))
            self.problem.logger.debug(f"Expansion Tree:\n{tree_str}")

            for candidate, complexity in self.expand(grammar, 'S', 0):
                simplified_expr = self.simplify_term(candidate)
                expr_str = str(simplified_expr)
                if expr_str not in self.explored_expressions[func_name]:
                    self.problem.logger.debug(f'Generated candidate: {expr_str} (complexity: {complexity})')
                    self.explored_expressions[func_name].add(expr_str)
                    candidates.append((simplified_expr, func_name))
                    break

        return candidates

    def expand(self, grammar, expr, depth):
        if depth > self.max_depth:
            return []
        expansions = []

        if isinstance(expr, tuple):
            if len(expr) == 2 and isinstance(expr[1], int):
                symbol, weight = expr
            else:
                symbol, weight = expr, 0
        else:
            symbol, weight = expr, 0

        if symbol in grammar:
            # trying complex first, may make this configurable
            sorted_productions = sorted(grammar[symbol], key=lambda x: -x[1])
            for production, prod_weight in sorted_productions:
                new_weight = weight + prod_weight
                sub_expansions = self.expand(grammar, production, depth + 1)
                expansions.extend([(exp, w + new_weight) for exp, w in sub_expansions])
        elif isinstance(symbol, str):
            return [(z3.Int(symbol), weight)]
        elif isinstance(symbol, tuple):
            op, *args = symbol
            arg_expansions = [self.expand(grammar, arg, depth + 1) for arg in args]

            for arg_combo in product(*arg_expansions):
                args = [arg[0] for arg in arg_combo]
                new_weight = sum(arg[1] for arg in arg_combo) + weight
                if op == '+':
                    expansions.append((args[0] + args[1], new_weight))
                elif op == '-':
                    expansions.append((args[0] - args[1], new_weight))
                elif op == '*':
                    if isinstance(args[0], z3.IntNumRef) or isinstance(args[1], z3.IntNumRef):
                        expansions.append((args[0] * args[1], new_weight))
                elif op == 'ite':
                    expansions.append((z3.If(args[0], args[1], args[2]), new_weight))
                elif op == '>':
                    expansions.append((args[0] > args[1], new_weight))
                elif op == '>=':
                    expansions.append((args[0] >= args[1], new_weight))
                elif op == '<=':
                    expansions.append((args[0] <= args[1], new_weight))
                elif op == '<':
                    expansions.append((args[0] < args[1], new_weight))
                elif op == '==':
                    expansions.append((args[0] == args[1], new_weight))
                elif op == '!=':
                    expansions.append((args[0] != args[1], new_weight))
                elif op == 'Neg':
                    expansions.append((-args[0], new_weight))

        return sorted(expansions, key=lambda x: -x[1])  # Sort expansions by weight in descending order

    def simplify_term(self, term: Union[z3.ExprRef, int]) -> Union[z3.ExprRef, int]:
        if isinstance(term, z3.ExprRef):
            return z3.simplify(term)
        return term

    def prune_candidates(self, candidates: List[Tuple[z3.ExprRef, str]]) -> List[Tuple[z3.ExprRef, str]]:
        return candidates

    def build_tree(self, grammar, expr, depth=0):
        if isinstance(expr, tuple):
            if len(expr) == 2 and isinstance(expr[1], int):
                symbol, weight = expr
            else:
                symbol, weight = expr, 0
        else:
            symbol, weight = expr, 0

        node = Node(str(symbol), weight)
        if depth > 10:
            return node

        if symbol in grammar:
            for production, prod_weight in grammar[symbol]:
                child = self.build_tree(grammar, production, depth + 1)
                child.weight = prod_weight
                node.children.append(child)
        elif isinstance(symbol, tuple):
            op, *args = symbol
            for arg in args:
                child = self.build_tree(grammar, arg, depth + 1)
                node.children.append(child)

        return node

    def print_tree(self, node, prefix="", is_last=True):
        result = [prefix + ("└── " if is_last else "├── ") + f"{node.value} (weight: {node.weight})"]
        for i, child in enumerate(node.children):
            result.extend(self.print_tree(child, prefix + ("    " if is_last else "│   "), i == len(node.children) - 1))
        return result
