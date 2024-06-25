import random
from typing import List, Tuple

from z3 import *

from src.cegis.z3.synthesis_problem import SynthesisProblem
from src.cegis.z3.synthesis_strategy import SynthesisStrategy


class RandomSearchStrategyBottomUp(SynthesisStrategy):
    def __init__(self, problem: SynthesisProblem):
        super().__init__(problem)
        self.problem = problem
        self.min_const = problem.options.min_const
        self.max_const = problem.options.max_const
        self.term_bank = {}

    def generate_random_term(self, arg_sorts: List[z3.SortRef], depth: int, complexity: int,
                             operations: List[str] = None) -> Tuple[z3.ExprRef, str]:
        if operations is None:
            operations = ['+', '-', '*', 'If', 'Neg']

        args = [z3.Var(i, sort) for i, sort in enumerate(arg_sorts)]
        num_args = len(args)
        constants = [z3.IntVal(i) for i in range(self.min_const, self.max_const + 1)]

        def build_term(curr_depth: int, curr_complexity: int) -> z3.ExprRef:
            if curr_depth == 0 or curr_complexity == 0:
                return random.choice(args + constants)

            available_ops = [op for op in operations if curr_complexity >= self.op_complexity(op)]
            if not available_ops:
                return random.choice(args + constants)

            op = random.choice(available_ops)
            remaining_complexity = curr_complexity - self.op_complexity(op)

            if op in ['+', '-']:
                left = build_term(curr_depth - 1, remaining_complexity // 2)
                right = build_term(curr_depth - 1, remaining_complexity - (remaining_complexity // 2))
                return left + right if op == '+' else left - right
            elif op == '*':
                left = random.choice(args) if args else random.choice(constants)
                right = random.choice(constants)
                return left * right
            elif op == 'If':
                condition = self.generate_condition(args)
                true_expr = build_term(curr_depth - 1, remaining_complexity // 2)
                false_expr = build_term(curr_depth - 1, remaining_complexity - (remaining_complexity // 2))
                return z3.If(condition, true_expr, false_expr)
            elif op == 'Neg':
                return -build_term(curr_depth - 1, remaining_complexity)

        generated_expression = build_term(depth, complexity)
        self.problem.print_msg(f"Generated expression: {generated_expression}", level=1)

        func_str = f"def arithmetic_function({', '.join(f'arg{i}' for i in range(num_args))}):\n"
        func_str += f"    return {str(generated_expression)}\n"

        return generated_expression, func_str

    def generate_condition(self, args: List[z3.ExprRef]) -> z3.BoolRef | bool:
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

    @staticmethod
    def op_complexity(op: str) -> int:
        # experimenting with cost of operation for biasing random choice
        return {'+': 1, '-': 1, '*': 2, 'If': 3, 'Neg': 1}.get(op, 0)

    def generate_candidates(self) -> List[Tuple[z3.ExprRef, str]]:
        candidates = []
        func_strs = []

        for func_name, arg_sorts in self.problem.context.variable_mapping_dict.items():
            candidate, func_str = self.generate_random_term(list(arg_sorts.values()),
                                                            self.problem.options.max_depth,
                                                            self.problem.options.max_complexity)
            candidates.append((candidate, func_name))
            func_strs.append(func_str)

        return candidates

    def prune_candidates(self, candidates: List[Tuple[z3.ExprRef, str]]) -> List[Tuple[z3.ExprRef, str]]:
        # no pruning to see here
        return candidates

    def execute_cegis(self) -> None:
        max_depth = self.problem.options.max_depth
        max_complexity = self.problem.options.max_complexity
        max_candidates_per_depth = self.problem.options.max_candidates_at_each_depth

        for depth in range(1, max_depth + 1):
            for complexity in range(1, max_complexity + 1):
                for iteration in range(max_candidates_per_depth):
                    candidates = self.generate_candidates()
                    pruned_candidates = self.prune_candidates(candidates)

                    self.problem.print_msg(
                        f"Testing candidates (depth: {depth}, complexity: {complexity}, iteration: {iteration + 1}):\n{'; '.join([func_str for _, func_str in candidates])}",
                        level=1
                    )

                    if self.test_candidates(pruned_candidates):
                        self.problem.print_msg(f"Found satisfying candidates!", level=2)
                        for candidate, func_name in pruned_candidates:
                            self.problem.print_msg(f"{func_name}: {candidate}", level=2)
                        self.set_solution_found()
                        return

        self.problem.print_msg("No satisfying candidates found.", level=2)
