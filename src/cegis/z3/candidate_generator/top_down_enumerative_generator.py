from itertools import product
from typing import List, Tuple, Union
import z3
from src.cegis.z3.synthesis_problem import SynthesisProblem


class Node:
    def __init__(self, value):
        self.value = value
        self.children = []


class TopDownCandidateGenerator:
    """
    A top-down candidate generator for synthesis problems.

    This class generates candidate solutions for synthesis problems using a top-down
    approach. It starts with complex expressions and gradually simplifies them,
    exploring the solution space in a top-down manner.

    Attributes:
        grammar (dict): The grammar used for generating expressions.
        problem (SynthesisProblem): The synthesis problem being solved.
        min_const (int): The minimum constant value to be used in expressions.
        max_const (int): The maximum constant value to be used in expressions.
        max_depth (int): The maximum depth of generated expressions.
        explored_expressions (dict): A dictionary to keep track of explored expressions for each function.

    Methods:
        define_grammar(variables): Define the grammar for expression generation.
        generate_candidates(): Generate candidate expressions.
        expand(grammar, expr, depth): Expand an expression using the grammar.
        simplify_term(term): Simplify a term in the expression.
        prune_candidates(candidates): Prune the list of candidate expressions.
        build_tree(grammar, expr, depth): Build a tree representation of the grammar.
        print_tree(node, prefix, is_last): Print the tree representation of the grammar.
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
        Define the grammar for expression generation.

        If no custom grammar is provided, this method creates a default grammar
        using the given variables and the configured constant range.

        Args:
            variables (list): List of variable names to be used in the grammar.

        Returns:
            dict: The defined grammar.
        """
        if self.grammar is None:
            return {
                'S': ['T', ('ite', 'B', 'S', 'S'), ('+', 'S', 'S'), ('-', 'S', 'S'), ('*', 'S', 'S'), ('neg', 'S')],
                'B': [('>', 'T', 'T'), ('>=', 'T', 'T'), ('<', 'T', 'T'), ('<=', 'T', 'T'), ('==', 'T', 'T'),
                      ('!=', 'T', 'T')],
                'T': list(variables) + [str(i) for i in range(self.min_const, self.max_const + 1)]
            }
        return self.grammar

    def generate_candidates(self) -> List[Tuple[z3.ExprRef, str]]:
        """
        Generate candidate expressions for each function in the synthesis problem.

        This method generates candidate expressions using the defined grammar,
        simplifies them, and ensures that only unexplored expressions are considered.

        Returns:
            List[Tuple[z3.ExprRef, str]]: A list of tuples, where each tuple contains
            a candidate expression and the name of the function it's for.
        """
        candidates = []
        for iteration, (func_name, variable_mapping) in enumerate(self.problem.context.variable_mapping_dict.items(),1):
            variables = [str(var) for var in variable_mapping.values()]
            grammar = self.define_grammar(variables)
            self.problem.logger.debug(f'Grammar {grammar}')
            self.problem.logger.debug(f"\nIteration {iteration}:")
            tree = self.build_tree(grammar, 'S')
            tree_str = "\n".join(self.print_tree(tree))
            self.problem.logger.debug(tree_str)
            for candidate in self.expand(grammar, 'S', 0):
                simplified_expr = self.simplify_term(candidate)
                expr_str = str(simplified_expr)
                if expr_str not in self.explored_expressions[func_name]:
                    self.problem.logger.info(f'Generated expression: {expr_str}')
                    self.explored_expressions[func_name].add(expr_str)
                    candidates.append((simplified_expr, func_name))
                    break

        return candidates

    def expand(self, grammar: dict[str, list], expr: str, depth: int) -> List[Tuple[z3.ExprRef, str]]:
        """
        Recursively expand an expression using the grammar.

        This method generates all possible expansions of a given expression
        according to the grammar rules, up to a maximum depth.

        Args:
            grammar (dict): The grammar to use for expansions.
            expr (Union[str, tuple]): The expression to expand.
            depth (int): The current depth in the expansion process.

        Returns:
            List[z3.ExprRef]: A list of expanded expressions.

        Example:
            If expr is 'S' and the grammar includes a rule 'S': ['+', 'T', 'T'],
            this method might return expansions like [x + y, x + 1, 1 + x, ...].
        """
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
                elif op == 'neg':
                    expansions.append(-arg_combo[0])
        return expansions

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

    def build_tree(self, grammar: dict[str, list], expr: Union[str, tuple], depth: int = 0):
        """
        Build a tree representation of the grammar.

        This method constructs a tree structure that represents the
        possible expansions of an expression according to the grammar.

        Args:
            grammar (dict): The grammar to use for tree construction.
            expr (Union[str, tuple]): The expression to expand into a tree.
            depth (int, optional): The current depth in the tree. Defaults to 0.

        Returns:
            Node: The root node of the constructed tree.

        Note:
            The tree construction stops at a depth of 10 to prevent
            infinite recursion for recursive grammars.
        """
        node = Node(str(expr))
        if depth > 10:
            return node

        if isinstance(expr, tuple):
            op, *args = expr
            for arg in args:
                child = self.build_tree(grammar, arg, depth + 1)
                node.children.append(child)
        elif expr in grammar:
            for production in grammar[expr]:
                child = self.build_tree(grammar, production, depth + 1)
                node.children.append(child)

        return node

    def print_tree(self, node: Node, prefix: str = "", is_last: bool = True):
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

                S
                ├── T
                │   ├── x
                │   └── y
                └── ('+', 'T', 'T')
                    ├── +
                    ├── T
                    │   ├── x
                    │   └── y
                    └── T
                        ├── x
                        └── y
        """
        result = [prefix + ("└── " if is_last else "├── ") + node.value]
        for i, child in enumerate(node.children):
            result.extend(self.print_tree(child, prefix + ("    " if is_last else "│   "), i == len(node.children) - 1))
        return result
