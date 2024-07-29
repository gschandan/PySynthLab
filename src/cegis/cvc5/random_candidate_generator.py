import random
from typing import List, Tuple
import cvc5

from src.cegis.cvc5.candidate_generator_base import CandidateGeneratorCVC5
from src.cegis.cvc5.synthesis_problem_cvc5 import SynthesisProblemCVC5


# TODO refactor/abstract to allow swapping out z3 for cvc5
class RandomCandidateGeneratorCVC5(CandidateGeneratorCVC5):
    def __init__(self, problem: 'SynthesisProblemCVC5'):
        super().__init__(problem)
        self.solver = problem.solver

    def generate_candidates(self) -> List[Tuple[cvc5.Term, str]]:
        candidates = []
        for func_name, variable_mapping in self.problem.cvc5_variables.items():
            candidate = self.generate_random_term(
                self.get_arg_sorts(func_name),
                self.problem.options.synthesis_parameters.max_depth,
                self.problem.options.synthesis_parameters.max_complexity,
            )
            candidates.append((candidate, func_name))
        return candidates

    def generate_random_term(self, arg_sorts: List[cvc5.Sort], depth: int, complexity: int,
                             operations: List[str] = None) -> cvc5.Term:
        if operations is None:
            operations = ['+', '-', '*', 'ite', 'neg']

        args = [self.solver.mkVar(sort) for sort in arg_sorts]
        constants = [self.solver.mkInteger(i) for i in range(self.min_const, self.max_const + 1)]

        op_weights = {op: self.op_complexity(op) for op in operations}

        def build_term(curr_depth: int, curr_complexity: int) -> cvc5.Term:
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
                return self.solver.mkTerm(cvc5.Kind.ADD if op == '+' else cvc5.Kind.SUB, left, right)
            elif op == '*':
                left = random.choice(args) if args else random.choice(constants)
                right = random.choice(constants)
                return self.solver.mkTerm(cvc5.Kind.MULT, left, right)
            elif op == 'ite':
                condition = self.generate_condition(args)
                true_expr = build_term(curr_depth - 1, remaining_complexity // 2)
                false_expr = build_term(curr_depth - 1, remaining_complexity - (remaining_complexity // 2))
                return self.solver.mkTerm(cvc5.Kind.ITE, condition, true_expr, false_expr)
            elif op == 'neg':
                return self.solver.mkTerm(cvc5.Kind.NEG, build_term(curr_depth - 1, remaining_complexity))
            self.problem.logger.error(f"Unexpected operation: {op}")
            raise ValueError(f"Unexpected operation: {op}")

        generated_expression = build_term(depth, complexity)
        self.problem.logger.info(f"Generated expression: {generated_expression}")

        return generated_expression

    def generate_condition(self, args: List[cvc5.Term]) -> cvc5.Term:
        comparisons = [cvc5.Kind.LT, cvc5.Kind.LEQ, cvc5.Kind.GT, cvc5.Kind.GEQ, cvc5.Kind.EQUAL, cvc5.Kind.DISTINCT]
        left = random.choice(args)
        right = random.choice(args + [self.solver.mkInteger(random.randint(self.min_const, self.max_const))])
        op = random.choice(comparisons)
        return self.solver.mkTerm(op, left, right)

    def prune_candidates(self, candidates: List[Tuple[cvc5.Term, str]]) -> List[Tuple[cvc5.Term, str]]:
        return candidates