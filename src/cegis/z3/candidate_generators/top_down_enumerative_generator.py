import random
from src.cegis.z3.candidate_generators.candidate_generator_base import CandidateGenerator
from typing import List, Tuple
from z3 import *


class TopDownCandidateGenerator(CandidateGenerator):

    def prune_candidates(self, candidates: List[Tuple[z3.ExprRef, str]]) -> List[Tuple[z3.ExprRef, str]]:
        return candidates

    def generate_candidates(self) -> List[Tuple[z3.ExprRef, str]]:
        candidates = []
        for func_name, variable_mapping in self.problem.context.variable_mapping_dict.items():
            candidate = self.generate_random_term(
                self.get_arg_sorts(func_name),
                self.config.synthesis_parameters_max_depth,
                self.config.synthesis_parameters_max_complexity
            )
            candidates.append((candidate, func_name))
        return candidates

    def generate_random_term(self, arg_sorts: List[z3.SortRef], max_depth: int, max_complexity: int,
                             operations: List[str] = None) -> z3.ExprRef:

        if operations is None:
            operations = ['+', '-', '*', 'If', 'Neg']
        args = [z3.Var(i, sort) for i, sort in enumerate(arg_sorts)]
        constants = [z3.IntVal(i) for i in range(self.min_const, self.max_const + 1)]

        op_weights = {op: 1 for op in operations}

        def build_term(curr_depth: int, remaining_complexity: int) -> z3.ExprRef:
            if curr_depth == 0 or remaining_complexity <= 0:
                return random.choice(args + constants)

            available_ops = [op for op in operations if remaining_complexity >= self.op_complexity(op)]
            if not available_ops:
                return random.choice(args + constants)

            op = random.choices(available_ops, weights=[op_weights[op] for op in available_ops])[0]
            new_complexity = remaining_complexity - self.op_complexity(op)

            op_weights[op] *= 0.9

            if op in ['+', '-']:
                left = build_term(curr_depth - 1, new_complexity)
                right = build_term(curr_depth - 1, new_complexity)
                return left + right if op == '+' else left - right
            elif op == '*':
                left = build_term(curr_depth - 1, new_complexity)
                right = random.choice(constants)
                return left * right
            elif op == 'If':
                condition = self.generate_condition(args)
                true_expr = build_term(curr_depth - 1, new_complexity)
                false_expr = build_term(curr_depth - 1, new_complexity)
                return z3.If(condition, true_expr, false_expr)
            elif op == 'Neg':
                return -build_term(curr_depth - 1, new_complexity)

        generated_expression = build_term(max_depth, max_complexity)
        self.problem.print_msg(f"Generated expression: {generated_expression}", level=1)
        return generated_expression

    def generate_condition(self, arg_sorts: List[z3.SortRef]) -> z3.BoolRef:
        operands = [random.choice(arg_sorts + [IntSort()]) for _ in range(2)]
        op = random.choice(['<', '<=', '>', '>=', '==', '!='])

        left = z3.Var(0, operands[0])
        right = z3.Var(1, operands[1])
        return {
            '<': left < right,
            '<=': left <= right,
            '>': left > right,
            '>=': left >= right,
            '==': left == right,
            '!=': left != right
        }[op]

    @staticmethod
    def op_complexity(op: str) -> int:
        # experimenting with cost of operation for biasing random choice
        return {'+': 1, '-': 1, '*': 2, 'If': 3, 'Neg': 1}.get(op, 0)