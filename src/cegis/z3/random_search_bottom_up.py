import random
from typing import List, Tuple, Dict
from z3 import *
from src.cegis.z3.synthesis_problem import SynthesisProblem
from src.cegis.z3.synthesis_strategy import SynthesisStrategy


class RandomSearchStrategyBottomUp(SynthesisStrategy):
    def __init__(self, problem: SynthesisProblem):
        super().__init__()
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

    def execute_cegis(self) -> None:
        max_depth = self.problem.options.max_depth
        max_complexity = self.problem.options.max_complexity
        max_candidates_per_depth = self.problem.options.max_candidates_at_each_depth
        counterexamples = {}

        arg_sorts = [x.sort() for x in list(self.problem.context.variable_mapping_dict.values())[0].keys()]

        for depth in range(1, max_depth + 1):
            for complexity in range(1, max_complexity + 1):
                for iteration in range(max_candidates_per_depth):
                    candidate, func_str = self.generate_random_term(arg_sorts, depth, complexity)

                    self.problem.print_msg(
                        f"Testing candidate (depth: {depth}, complexity: {complexity}, iteration: {iteration + 1}):\n{func_str}",
                        level=1)

                    if not self.check_counterexamples([(candidate, 'f')], counterexamples):
                        continue

                    if self.verify_candidates([candidate]):
                        new_counterexamples = self.generate_counterexample([(candidate, 'f')])
                        if new_counterexamples is None:
                            self.problem.print_msg(f"Found satisfying candidate!", level=2)
                            self.problem.print_msg(f"f: {candidate}", level=2)
                            self.set_solution_found()
                            return
                        else:
                            counterexamples.update(new_counterexamples)
                            for func_name, ce in new_counterexamples.items():
                                self.problem.print_msg(
                                    f"New counterexample found: {func_name}({', '.join([str(val) for val in ce.values()])})",
                                    level=1)
                    else:
                        self.problem.print_msg("Candidate failed verification.", level=1)

        self.problem.print_msg("No satisfying candidates found.", level=2)

    def check_counterexamples(self, candidates: List[Tuple[z3.ExprRef, str]],
                              counterexamples: Dict[str, Dict[str, int]]) -> bool:
        for (candidate, func_name) in candidates:
            if func_name not in counterexamples:
                continue
            ce = counterexamples[func_name]
            variable_mapping = self.problem.context.variable_mapping_dict[func_name]
            substituted_expr = z3.substitute(candidate, [
                (var, z3.IntVal(ce[str(var)])) for var in variable_mapping.keys()
            ])
            result = z3.simplify(substituted_expr)
            if not self.satisfies_constraints(func_name, candidate, result):
                return False
        return True

    def verify_candidates(self, candidates: List[z3.ExprRef]) -> bool:
        self.problem.context.verification_solver.reset()
        if self.problem.options.randomise_each_iteration:
            self.problem.context.verification_solver.set('random_seed', random.randint(1, 4294967295))

        substituted_constraints = self.problem.substitute_constraints(
            self.problem.context.z3_constraints,
            list(self.problem.context.z3_synth_functions.values()),
            candidates)
        self.problem.context.verification_solver.add(substituted_constraints)

        return self.problem.context.verification_solver.check() == z3.sat

    def generate_counterexample(self, candidates: List[Tuple[z3.ExprRef, str]]) -> dict[str, dict[str, int]] | None:
        self.problem.context.enumerator_solver.reset()
        if self.problem.options.randomise_each_iteration:
            self.problem.context.enumerator_solver.set('random_seed', random.randint(1, 4294967295))

        substituted_neg_constraints = self.problem.substitute_constraints(
            self.problem.context.z3_negated_constraints,
            list(self.problem.context.z3_synth_functions.values()),
            [candidate for candidate, _ in candidates])
        self.problem.context.enumerator_solver.add(substituted_neg_constraints)

        if self.problem.context.enumerator_solver.check() == z3.sat:
            model = self.problem.context.enumerator_solver.model()
            counterexamples = {}

            for (candidate, func_name) in candidates:
                variable_mapping = self.problem.context.variable_mapping_dict[func_name]
                counterexample = {str(free_var): model.eval(declared_var, model_completion=True).as_long()
                                  for free_var, declared_var in variable_mapping.items()}

                incorrect_output = z3.simplify(z3.substitute(candidate, [
                    (arg, z3.IntVal(value)) for arg, value in zip(variable_mapping.keys(), counterexample.values())
                ]))

                self.problem.print_msg(f"Counterexample for {func_name}: {counterexample}", level=0)
                counterexamples[func_name] = counterexample
                self.problem.print_msg(f"Incorrect output for {func_name}: {incorrect_output}", level=0)
                self.problem.context.counterexamples.append((func_name, counterexample, incorrect_output))

            return counterexamples

        return None

    def satisfies_constraints(self, func_name: str, candidate: z3.ExprRef, result: z3.ExprRef) -> bool:
        solver = z3.Solver()
        substituted_constraints = self.problem.substitute_constraints(
            self.problem.context.z3_constraints,
            [self.problem.context.z3_synth_functions[func_name]],
            [candidate])
        solver.add(substituted_constraints)
        solver.add(self.problem.context.z3_synth_functions[func_name](
            *self.problem.context.variable_mapping_dict[func_name].values()) == result)
        return solver.check() == z3.sat
