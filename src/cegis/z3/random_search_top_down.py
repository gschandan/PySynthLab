import random
from typing import List, Tuple, Dict

import z3

from src.cegis.z3.synthesis_problem import SynthesisProblem
from src.cegis.z3.synthesis_strategy import SynthesisStrategy


class RandomSearchStrategyTopDown(SynthesisStrategy):
    def __init__(self, problem: SynthesisProblem):
        super().__init__(problem)
        self.problem = problem
        self.min_const = problem.options.min_const
        self.max_const = problem.options.max_const

    def generate_random_term(self, arg_sorts: List[z3.SortRef], depth: int, complexity: int,
                             operations: List[str] = None) -> z3.ExprRef:
        if operations is None:
            operations = ['+', '-', '*', 'If', 'Neg']

        args = [z3.Var(i, sort) for i, sort in enumerate(arg_sorts)]
        num_args = len(args)

        def build_term(curr_depth: int, curr_complexity: int) -> z3.ExprRef:
            if curr_depth == 0 or curr_complexity == 0:
                return random.choice(args) if args else z3.IntVal(random.randint(self.min_const, self.max_const))

            op = random.choice(operations)
            if op == 'If':
                condition = self.generate_condition(args)
                true_expr = build_term(curr_depth - 1, curr_complexity - 1)
                false_expr = build_term(curr_depth - 1, curr_complexity - 1)
                return z3.If(condition, true_expr, false_expr)
            elif op == 'Neg':
                expr = build_term(curr_depth - 1, curr_complexity - 1)
                return -expr
            elif op in ['+', '-']:
                left_expr = build_term(curr_depth - 1, curr_complexity - 1)
                right_expr = build_term(curr_depth - 1, curr_complexity - 1)
                if op == '+':
                    return left_expr + right_expr
                elif op == '-':
                    return left_expr - right_expr
            elif op == '*':
                left_expr = random.choice(args) if args else z3.IntVal(random.randint(self.min_const, self.max_const))
                right_expr = z3.IntVal(random.randint(self.min_const, self.max_const))
                return left_expr * right_expr

        generated_expression = build_term(depth, complexity)

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

    def generate_candidates(self) -> Dict[z3.ExprRef, str]:
        candidates = {}
        for func_name, variable_mapping in self.problem.context.variable_mapping_dict.items():
            depth = random.randint(1, self.problem.options.max_depth)
            complexity = random.randint(1, self.problem.options.max_complexity)
            candidate = self.generate_random_term([x.sort() for x in list(variable_mapping.keys())], depth, complexity)
            candidates[candidate] = func_name
        return candidates

    def prune_candidates(self, candidates: Dict[z3.ExprRef, str]) -> Dict[z3.ExprRef, str]:
        # no pruning atm
        return candidates

    def execute_cegis(self) -> None:
        max_iterations = self.problem.options.max_candidates_at_each_depth

        for iteration in range(max_iterations):
            candidates = self.prune_candidates(self.generate_candidates())

            self.problem.print_msg(f"Testing candidates (iteration {iteration + 1}):", level=1)

            if self.test_candidates_old(candidates):
                self.problem.print_msg("-" * 100, level=2)
                self.problem.print_msg(f"Found satisfying candidates!", level=2)
                for candidate, func_name in candidates:
                    self.problem.print_msg(f"{func_name}: {candidate}", level=2)
                self.set_solution_found()
                return

        self.problem.print_msg("No satisfying candidates found.", level=2)
