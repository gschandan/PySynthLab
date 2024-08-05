import logging
from itertools import product
from typing import List, Tuple, Union
import z3


class Node:
    def __init__(self, value, weight=0):
        self.value = value
        self.weight = weight
        self.children = []


class WeightedTopDownCandidateGenerator:
    """
    A weighted top-down candidate generator for synthesis problems.

    This class generates candidate solutions for synthesis problems using a weighted
    top-down approach. It assigns weights to different productions in the grammar,
    allowing for more control over the generation process.

    Attributes:
        grammar (dict): The weighted grammar used for generating expressions.
        problem (SynthesisProblem): The synthesis problem being solved.
        min_const (int): The minimum constant value to be used in expressions.
        max_const (int): The maximum constant value to be used in expressions.
        max_depth (int): The maximum depth of generated expressions.
        explored_expressions (dict): A dictionary to keep track of explored expressions for each function.

    Methods:
        define_grammar(variables): Define the weighted grammar for expression generation.
        generate_candidates(): Generate candidate expressions using the weighted grammar.
        expand(grammar, expr, depth): Expand an expression using the weighted grammar.
        simplify_term(term): Simplify a term in the expression.
        prune_candidates(candidates): Prune the list of candidate expressions.
        build_tree(grammar, expr, depth): Build a tree representation of the weighted grammar.
        print_tree(node, prefix, is_last): Print the tree representation of the weighted grammar.
    """

    def __init__(self, problem: 'SynthesisProblem'):
        self.grammar = None
        self.problem = problem
        self.min_const = problem.options.synthesis_parameters.min_const
        self.max_const = problem.options.synthesis_parameters.max_const
        self.max_depth = problem.options.synthesis_parameters.max_depth
        self.explored_expressions: dict[str, set[str]] = {func_name: set() for func_name in
                                                          problem.context.variable_mapping_dict.keys()}
        self.grammar = problem.options.synthesis_parameters.custom_grammar

    def define_grammar(self, variables: list[str]) -> dict[str, list]:
        """
        Define the weighted grammar for expression generation.

        If no custom grammar is provided, this method creates a default weighted grammar
        using the given variables and the configured constant range.

        Args:
            variables (list): List of variable names to be used in the grammar.

        Returns:
            dict: The defined weighted grammar.
        """
        if self.grammar is None:
            return {
                'S': [
                    (('ite', 'B', 'S', 'S'), 50),
                    (('+', 'S', 'S'), 30),
                    (('-', 'S', 'S'), 30),
                    (('*', 'S', 'S'), 35),
                    (('neg', 'S'), 20),
                    ('T', 10),
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
        return self.grammar

    def generate_candidates(self) -> List[Tuple[z3.ExprRef, str]]:
        """
        Generate candidate expressions for each function in the synthesis problem.

        This method generates candidate expressions using the defined weighted grammar,
        simplifies them, and ensures that only unexplored expressions are considered.
        It also takes into account the weights of different productions.

        Returns:
            List[Tuple[z3.ExprRef, str]]: A list of tuples, where each tuple contains
            a candidate expression and the name of the function it's for.
        """
        candidates = []
        for func_name, variable_mapping in self.problem.context.variable_mapping_dict.items():
            variables = [str(var) for var in variable_mapping.values()]
            grammar = self.define_grammar(variables)
            self.problem.logger.debug(f'Grammar {grammar}')

            if self.problem.logger.getEffectiveLevel() <= logging.DEBUG:
                tree = self.build_tree(grammar, 'S')
                tree_str = "\n".join(self.print_tree(tree))
                self.problem.logger.debug(tree_str)

            for candidate, complexity in self.expand(grammar, 'S', 0):
                simplified_expr = self.simplify_term(candidate)
                expr_str = str(simplified_expr)
                if expr_str not in self.explored_expressions[func_name]:
                    self.problem.logger.debug(f'Generated candidate: {expr_str} (complexity: {complexity})')
                    self.explored_expressions[func_name].add(expr_str)
                    candidates.append((simplified_expr, func_name))
                    break

        return candidates

    def expand(self, grammar: dict[str, list], expr: str, depth: int) -> list[tuple[z3.ExprRef, str]]:
        """
        Recursively expand an expression using the weighted grammar.

        This method generates all possible expansions of a given expression
        according to the grammar rules, up to a maximum depth. It also
        calculates and propagates weights for each expansion.

        Args:
            grammar (dict): The weighted grammar to use for expansions.
            expr (Union[str, tuple]): The expression to expand.
            depth (int): The current depth in the expansion process.

        Returns:
            List[Tuple[z3.ExprRef, int]]: A list of tuples, each containing
            an expanded expression and its associated weight, sorted by
            weight in descending order.

        Example:
            If expr is 'S' and the grammar includes a rule 'S': [('+', 'T', 'T'), 30],
            this method might return expansions like [(x + y, 35), (x + 1, 40), ...].
        """
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
                elif op == 'neg':
                    expansions.append((-args[0], new_weight))

        return sorted(expansions, key=lambda x: -x[1])

    @staticmethod
    def simplify_term(term: Union[z3.ExprRef, int]) -> Union[z3.ExprRef, int]:
        """
        Simplify a term in the expression.

        This method uses Z3's simplification for Z3 expressions and
        returns integers as-is.

        Args:
            term (Union[z3.ExprRef, int]): The term to simplify.

        Returns:
            Union[z3.ExprRef, int]: The simplified term.

        Example:
            simplify_term(z3.Int('x') + 0) might return z3.Int('x')
            simplify_term(5) would return 5
        """
        if isinstance(term, z3.ExprRef):
            return z3.simplify(term)
        return term

    @staticmethod
    def prune_candidates(candidates: List[Tuple[z3.ExprRef, str]]) -> List[Tuple[z3.ExprRef, str]]:
        """
        Prune the list of candidate expressions.

        This method is a placeholder for potential future pruning strategies.
        Currently, it returns the candidates unchanged.

        Args:
            candidates (List[Tuple[z3.ExprRef, str]]): The list of candidate
            expressions to prune.

        Returns:
            List[Tuple[z3.ExprRef, str]]: The pruned list of candidates.
        """
        return candidates

    def build_tree(self, grammar: dict[str, list], expr: str, depth: int = 0) -> Node:
        """
        Build a tree representation of the weighted grammar.

        This method constructs a tree structure that represents the
        possible expansions of an expression according to the grammar.

        Args:
            grammar (dict): The weighted grammar to use for tree construction.
            expr (Union[str, tuple]): The expression to expand into a tree.
            depth (int, optional): The current depth in the tree. Defaults to 0.

        Returns:
            Node: The root node of the constructed tree.

        Note:
            The tree construction stops at a depth of 10 to prevent
            infinite recursion for recursive grammars.
        """
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

    def print_tree(self, node, prefix="", is_last=True) -> list[str]:
        """
        Generate a string representation of the grammar tree.

        This method creates a formatted string that visually represents
        the structure of the grammar tree.

        Args:
            node (Node): The current node in the tree.
            prefix (str, optional): The prefix to use for this line. Defaults to "".
            is_last (bool, optional): Whether this is the last child of its parent. Defaults to True.

        Returns:
            List[str]: A list of strings, each representing a line in the tree visualization.

        Example:
            For a simple grammar tree, the output might look like:

            .. code-block::

                S (weight: 0)
                ├── T (weight: 10)
                │   ├── x (weight: 5)
                │   └── y (weight: 5)
                └── ('+', 'T', 'T') (weight: 30)
                    ├── + (weight: 0)
                    ├── T (weight: 10)
                    │   ├── x (weight: 5)
                    │   └── y (weight: 5)
                    └── T (weight: 10)
                        ├── x (weight: 5)
                        └── y (weight: 5)
        """
        result = [prefix + ("└── " if is_last else "├── ") + f"{node.value} (weight: {node.weight})"]
        for i, child in enumerate(node.children):
            result.extend(self.print_tree(child, prefix + ("    " if is_last else "│   "), i == len(node.children) - 1))
        return result
