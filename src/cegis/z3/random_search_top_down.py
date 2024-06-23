import random
from typing import List, Tuple, Dict
import z3

from src.cegis.z3.synthesis_problem import SynthesisProblem
from src.cegis.z3.synthesis_strategy import SynthesisStrategy


class RandomSearchStrategyTopDown(SynthesisStrategy):
    def __init__(self, problem: SynthesisProblem):
        super().__init__()
        self.problem = problem
        self.min_const = problem.options.min_const
        self.max_const = problem.options.max_const

    def generate_random_term(self, arg_sorts: List[z3.SortRef], depth: int, complexity: int,
                             operations: List[str] = None) -> Tuple[z3.ExprRef, str]:
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

    def execute_cegis(self) -> None:
        """Execute counterexample-guided inductive synthesis."""
        max_depth = self.problem.options.max_depth
        max_complexity = self.problem.options.max_complexity
        max_iterations = self.problem.options.max_candidates_at_each_depth

        counterexamples = {}

        for iteration in range(max_iterations):
            candidates = []
            func_strs = []

            for func_name, variable_mapping in self.problem.context.variable_mapping_dict.items():
                depth = random.randint(1, max_depth)
                complexity = random.randint(1, max_complexity)
                candidate, func_str = self.generate_random_term(
                    [x.sort() for x in list(variable_mapping.keys())], depth, complexity)
                candidates.append((candidate, func_name))
                func_strs.append(func_str)

            self.problem.print_msg(f"Testing candidates (iteration {iteration + 1}): {'; '.join(func_strs)}", level=1)

            if not self.check_counterexamples(candidates, counterexamples):
                continue

            if self.verify_candidates([c for c, _ in candidates]):
                new_counterexamples = self.generate_counterexample(candidates)
                if new_counterexamples is None:
                    self.problem.print_msg(f"Found satisfying candidates!", level=2)
                    for _, func_name in candidates:
                        self.problem.print_msg(
                            f"{func_name}: {self.problem.context.z3_synth_functions[func_name]}", level=2
                        )
                    self.set_solution_found()
                    return
                else:
                    counterexamples.update(new_counterexamples)
                    for func_name, ce in new_counterexamples.items():
                        self.problem.print_msg(
                            f"New counterexamples found: {func_name}({', '.join([str(val) for val in ce.values()])})",
                            level=1)
            else:
                self.problem.print_msg("Candidate failed verification.", level=1)

        self.problem.print_msg("No satisfying candidates found.", level=2)

    def check_counterexamples(self, candidates: List[Tuple[z3.ExprRef, str]],
                              counterexamples: Dict[str, Dict[str, int]]) -> bool:
        """Check if candidates satisfy all counterexamples."""
        for (candidate, func_name) in candidates:
            if func_name not in counterexamples:
                continue

            if not self.satisfies_constraints(candidate):
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

            for (candidate, func_str), (func_name, variable_mapping) in zip(candidates,
                                                                            self.problem.context.variable_mapping_dict.items()):
                free_variables = list(variable_mapping.keys())
                counterexample = {str(free_var): model.eval(declared_var, model_completion=True).as_long()
                                  for free_var, declared_var in variable_mapping.items()}

                incorrect_output = z3.simplify(z3.substitute(candidate, [(arg, z3.IntVal(value)) for arg, value in
                                                                         zip(free_variables,
                                                                             list(counterexample.values()))]))

                self.problem.print_msg(f"Counterexample for {func_name}: {counterexample}", level=0)
                counterexamples[func_name] = counterexample
                self.problem.print_msg(f"Incorrect output for {func_name}: {incorrect_output}", level=0)
                self.problem.context.counterexamples.append((func_str, counterexample, incorrect_output))

            return counterexamples

        return None

    def satisfies_constraints(self, candidate: z3.ExprRef) -> bool:
        solver = z3.Solver()
        substituted_constraints = self.problem.substitute_constraints(
            self.problem.context.z3_constraints,
            list(self.problem.context.z3_synth_functions.values()),
            [candidate])
        solver.add(substituted_constraints)
        return solver.check() == z3.sat
