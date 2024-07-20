import random
from src.cegis.z3.candidate_generator.candidate_generator_base import CandidateGenerator
from typing import List, Tuple
from z3 import *

from src.cegis.z3.synthesis_problem import SynthesisProblem


class TopDownCandidateGenerator(CandidateGenerator):

    def prune_candidates(self, candidates: List[Tuple[z3.ExprRef, str]]) -> List[Tuple[z3.ExprRef, str]]:
        return candidates

    def generate_candidates(self) -> List[Tuple[z3.ExprRef, str]]:
        candidates = []
        for func_name, variable_mapping in self.problem.context.variable_mapping_dict.items():
            candidate = self.generate_random_term(
                self.get_arg_sorts(func_name),
                SynthesisProblem.options.synthesis_parameters.max_depth,
                SynthesisProblem.options.synthesis_parameters.max_complexity
            )
            candidates.append((candidate, func_name))
        return candidates

    def generate_random_term(self, arg_sorts: List[z3.SortRef], max_depth: int, max_complexity: int,
                             operations: List[str] = None) -> z3.ExprRef:

        if operations is None:
            operations = ['+', '-', '*', 'If', 'Neg']
        args = [z3.Var(i, sort) for i, sort in enumerate(arg_sorts)]
        constants = [z3.IntVal(i) for i in range(self.min_const, self.max_const + 1)]

        op_weights = {op: self.op_complexity(op) for op in operations}

        def build_term(curr_depth: int, remaining_complexity: int) -> z3.ExprRef:
            if curr_depth == 0 or remaining_complexity <= 0:
                return random.choice(args + constants)

            available_ops = [op for op in operations if remaining_complexity >= self.op_complexity(op)]
            if not available_ops:
                return random.choice(args + constants)

            op = random.choices(available_ops, weights=[op_weights[op] for op in available_ops])[0]
            new_complexity = remaining_complexity - self.op_complexity(op)

            op_weights[op] *= SynthesisProblem.options.synthesis_parameters.weight_multiplier

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
            SynthesisProblem.logger.error(f"Unexpected operation: {op}")
            raise ValueError(f"Unexpected operation: {op}")

        generated_expression = build_term(max_depth, max_complexity)
        SynthesisProblem.logger.info(f"Generated expression: {generated_expression}")
        return generated_expression

    def generate_condition(self, args: List[z3.ExprRef]) -> z3.BoolRef:
        if len(args) < 2:
            return z3.BoolVal(random.choice([True, False]))

        left: z3.ExprRef = random.choice(args)
        right: z3.ExprRef = random.choice(args)
        while right == left:
            right = random.choice(args)

        op = random.choice(['<', '<=', '>', '>=', '=='])
        if op == '<':
            return left < right
        elif op == '<=':
            return left <= right
        elif op == '>':
            return left > right
        elif op == '>=':
            return left >= right
        else:
            return left == right
