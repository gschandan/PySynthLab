import random
from src.cegis.z3.candidate_generator.candidate_generator_base import CandidateGenerator
from typing import List, Tuple, Any
from z3 import *

from src.cegis.z3.synthesis_problem_z3 import SynthesisProblemZ3
from src.utilities.cancellation_token import GlobalCancellationToken


# experimental
class GuidedCandidateGenerator(CandidateGenerator):

    def prune_candidates(self, candidates: List[Tuple[z3.ExprRef, str]]) -> List[Tuple[z3.ExprRef, str]]:
        return candidates

    def generate_candidates(self) -> List[Tuple[z3.ExprRef, str]]:
        candidates = []
        for func_name, variable_mapping in self.problem.context.variable_mapping_dict.items():
            candidate = self.generate_guided_term(
                self.get_arg_sorts(func_name),
                SynthesisProblemZ3.options.synthesis_parameters.max_depth,
                SynthesisProblemZ3.options.synthesis_parameters.max_complexity,
                #self.config.candidate_generator_cost_function
            )
            candidates.append((candidate, func_name))
        return candidates

    def generate_guided_term(self, arg_sorts: List[z3.SortRef], depth: int, complexity: int,
                             cost_function: callable = None, operations: List[str] = None) -> z3.ExprRef:
        GlobalCancellationToken.check_cancellation()
        if operations is None:
            operations = ['+', '-', '*', 'ite', 'neg']

        args = [z3.Var(i, sort) for i, sort in enumerate(arg_sorts)]
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
            elif op == 'ite':
                condition = self.generate_condition(args)
                true_expr = build_term(curr_depth - 1, remaining_complexity // 2)
                false_expr = build_term(curr_depth - 1, remaining_complexity - (remaining_complexity // 2))
                return z3.If(condition, true_expr, false_expr)
            elif op == 'neg':
                return -build_term(curr_depth - 1, remaining_complexity)

        generated_expression = build_term(depth, complexity)
        SynthesisProblemZ3.logger.info(f"Generated expression: {generated_expression}")

        return generated_expression

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
        return {'+': 1, '-': 1, '*': 2, 'ite': 3, 'neg': 1}.get(op, 0)
