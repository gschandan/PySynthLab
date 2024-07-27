import uuid
from itertools import product
from typing import List, Tuple, Union
import z3
from treelib import Tree

from src.cegis.z3.synthesis_problem import SynthesisProblem


class Node:
    def __init__(self, value):
        self.value = value
        self.children = []


def build_tree(grammar, expr, depth=0):
    node = Node(str(expr))
    if depth > 10:
        return node

    if isinstance(expr, tuple):
        op, *args = expr
        for arg in args:
            child = build_tree(grammar, arg, depth + 1)
            node.children.append(child)
    elif expr in grammar:
        for production in grammar[expr]:
            child = build_tree(grammar, production, depth + 1)
            node.children.append(child)

    return node


def print_tree(node, prefix="", is_last=True):
    SynthesisProblem.logger.debug(prefix + ("└── " if is_last else "├── ") + node.value)
    for i, child in enumerate(node.children):
        print_tree(child, prefix + ("    " if is_last else "│   "), i == len(node.children) - 1)


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
            'S': ['T', ('ite', 'B', 'S', 'S'), ('+', 'S', 'S'), ('-', 'S', 'S'), ('*', 'S', 'S'), ('Neg', 'S')],
            'B': [('>', 'T', 'T'), ('>=', 'T', 'T'), ('<', 'T', 'T'), ('<=', 'T', 'T'), ('==', 'T', 'T'),
                  ('!=', 'T', 'T')],
            'T': list(variables) + [str(i) for i in range(self.min_const, self.max_const + 1)]
        }

    def generate_candidates(self) -> List[Tuple[z3.ExprRef, str]]:
        candidates = []
        for iteration, (func_name, variable_mapping) in enumerate(self.problem.context.variable_mapping_dict.items(), 1):
            variables = [str(var) for var in variable_mapping.values()]
            grammar = self.define_grammar(variables)
            self.problem.logger.debug(f'Grammar {grammar}')
            self.problem.logger.debug(f"\nIteration {iteration}:")
            tree = build_tree(grammar, 'S')
            print_tree(tree)
            for candidate in self.expand(grammar, 'S', 0):
                simplified_expr = self.simplify_term(candidate)
                expr_str = str(simplified_expr)
                if expr_str not in self.explored_expressions[func_name]:
                    self.problem.logger.debug(f'Generated candidate: {expr_str}')
                    self.explored_expressions[func_name].add(expr_str)
                    candidates.append((simplified_expr, func_name))
                    break

        return candidates

    def expand(self, grammar, expr, depth):
        if depth > self.max_depth:
            return []

        expansions = []

        if expr in grammar:
            for production in grammar[expr]:
                expansions.extend(self.expand(grammar, production, depth + 1))

        elif isinstance(expr, str):
            return [z3.Int(expr)]

        elif isinstance(expr, tuple):
            op, *args = expr
            arg_expansions = [self.expand(grammar, arg, depth + 1) for arg in args]

            for arg_combo in product(*arg_expansions):
                if op == '+':
                    expansions.append(arg_combo[0] + arg_combo[1])
                elif op == '-':
                    expansions.append(arg_combo[0] - arg_combo[1])
                elif op == '*':
                    if isinstance(args[0], z3.IntNumRef) or isinstance(args[1], z3.IntNumRef):
                        expansions.append((args[0] * args[1]))
                elif op == 'ite':
                    expansions.append(z3.If(arg_combo[0], arg_combo[1], arg_combo[2]))
                elif op == '>':
                    expansions.append(arg_combo[0] > arg_combo[1])
                elif op == '>=':
                    expansions.append(arg_combo[0] >= arg_combo[1])
                elif op == '<=':
                    expansions.append(arg_combo[0] <= arg_combo[1])
                elif op == '<':
                    expansions.append(arg_combo[0] < arg_combo[1])
                elif op == '==':
                    expansions.append(arg_combo[0] == arg_combo[1])
                elif op == '!=':
                    expansions.append(arg_combo[0] != arg_combo[1])
                elif op == 'Neg':
                    expansions.append(-arg_combo[0])
        return expansions

    def simplify_term(self, term: Union[z3.ExprRef, int]) -> Union[z3.ExprRef, int]:
        if isinstance(term, z3.ExprRef):
            return z3.simplify(term)
        return term

    def prune_candidates(self, candidates: List[Tuple[z3.ExprRef, str]]) -> List[Tuple[z3.ExprRef, str]]:
        return candidates
    