import random
from src.cegis.z3.candidate_generator.candidate_generator_base import CandidateGenerator
from typing import List, Tuple
from z3 import *


class RandomCandidateGenerator(CandidateGenerator):
    """
    A candidate generator that produces random expressions for synthesis problems.

    This class implements a bottom up strategy to generate random candidate solutions
    for synthesis problems. It creates expressions using a set of operations
    and adheres to specified complexity and depth constraints.

    Attributes:
        Inherits all attributes from the CandidateGenerator base class.
    """

    def generate_candidates(self) -> List[Tuple[z3.ExprRef, str]]:
        """
        Generate a list of random candidate expressions for each function in the synthesis problem.

        Returns:
            List[Tuple[z3.ExprRef, str]]: A list of tuples, where each tuple contains
            a randomly generated expression and the name of the function it's for.

        Example:
            If the problem has two functions 'f' and 'g', this method might return:
            [(x + 2, 'f'), (If(y > 0, y, -y), 'g')]
        """
        candidates = []
        for func_name, variable_mapping in self.problem.context.variable_mapping_dict.items():
            candidate = self.generate_random_term(
                self.get_arg_sorts(func_name),
                self.problem.options.synthesis_parameters.max_depth,
                self.problem.options.synthesis_parameters.max_complexity,
            )
            candidates.append((candidate, func_name))
        return candidates

    def generate_random_term(self, arg_sorts: List[z3.SortRef], depth: int, complexity: int,
                             operations: List[str] = None) -> z3.ExprRef:
        """
        Generate a random term (expression) based on given constraints.

        Args:
            arg_sorts (List[z3.SortRef]): List of argument sorts for the function.
            depth (int): Maximum depth of the generated expression tree.
            complexity (int): Maximum complexity allowed for the expression.
            operations (List[str], optional): List of allowed operations. Defaults to ['+', '-', '*', 'If', 'Neg'].

        Returns:
            z3.ExprRef: A randomly generated Z3 expression.

        Example:
            Given arg_sorts=[z3.IntSort(), z3.IntSort()], depth=3, complexity=5,
            this method might return an expression like: If(x > y, x + 2, y - 1)
        """
        if operations is None:
            operations = ['+', '-', '*', 'ite', 'neg']

        args = [z3.Var(i, sort) for i, sort in enumerate(arg_sorts)]
        constants = [z3.IntVal(i) for i in range(self.min_const, self.max_const + 1)]

        op_weights = {op: self.op_complexity(op) for op in operations}

        def build_term(curr_depth: int, curr_complexity: int) -> z3.ExprRef:
            if curr_depth == 0 or curr_complexity <= 0:
                return random.choice(args + constants)

            available_ops = [op for op in operations if curr_complexity >= self.op_complexity(op)]
            if not available_ops:
                return random.choice(args + constants)

            op = random.choices(available_ops, weights=[op_weights[op] for op in available_ops])[0]
            remaining_complexity = curr_complexity - self.op_complexity(op)

            op_weights[op] *= self.problem.options.synthesis_parameters.weight_multiplier

            if op in ['+', '-']:
                left = build_term(curr_depth - 1, remaining_complexity // 2)
                right = build_term(curr_depth - 1, remaining_complexity - (remaining_complexity // 2))
                return left + right if op == '+' else left - right
            elif op == '*':
                left = random.choice(args) if args else random.choice(constants)
                right = random.choice(constants)
                return left * right
            elif op == 'ite':
                condition = self.generate_condition(args)
                true_expr = build_term(curr_depth - 1, remaining_complexity // 2)
                false_expr = build_term(curr_depth - 1, remaining_complexity - (remaining_complexity // 2))
                return z3.If(condition, true_expr, false_expr)
            elif op == 'neg':
                return -build_term(curr_depth - 1, remaining_complexity)
            self.problem.logger.error(f"Unexpected operation: {op}")
            raise ValueError(f"Unexpected operation: {op}")

        generated_expression = build_term(depth, complexity)
        self.problem.logger.info(f"Generated expression: {generated_expression}")

        return generated_expression

    def generate_condition(self, args: List[z3.ExprRef]) -> z3.BoolRef | bool:

        """
        Generate a random boolean condition using the given arguments.

        Args:
            args (List[z3.ExprRef]): List of available arguments to use in the condition.

        Returns:
            z3.BoolRef | bool: A randomly generated boolean condition.

        Example:
            Given args=[x, y], this method might return conditions like:
            x < y, x == 2, y >= 0, etc.
        """
        comparisons = ['<', '<=', '>', '>=', '==', '!=']
        left = random.choice(args)
        right = random.choice(args + [z3.IntVal(random.randint(self.min_const, self.max_const))])
        op = random.choice(comparisons)

        if op == '<':
            return left < right
        elif op == '<=':
            return left <= right
        elif op == '>':
            return left > right
        elif op == '>=':
            return left >= right
        elif op == '==':
            return left == right
        else:
            return left != right

    def prune_candidates(self, candidates: List[Tuple[z3.ExprRef, str]]) -> List[Tuple[z3.ExprRef, str]]:
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
