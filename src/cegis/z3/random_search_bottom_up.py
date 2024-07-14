from typing import List

from z3 import SortRef, Var, If

from src.cegis.z3.synthesis_problem import SynthesisProblem
from src.cegis.z3.synthesis_strategy import SynthesisStrategy


class RandomSearchStrategyBottomUp(SynthesisStrategy):
    def __init__(self, problem: SynthesisProblem):
        super().__init__(problem)
        self.problem = problem
        self.min_const = problem.options.synthesis_parameters_min_const
        self.max_const = problem.options.synthesis_parameters_max_const
        self.term_bank = {}

    def execute_cegis(self) -> None:
        max_depth = self.problem.options.synthesis_parameters_max_depth
        max_complexity = self.problem.options.synthesis_parameters_max_complexity
        max_candidates_per_depth = self.problem.options.synthesis_parameters_max_candidates_at_each_depth

        for depth in range(1, max_depth + 1):
            for complexity in range(1, max_complexity + 1):
                for iteration in range(max_candidates_per_depth):
                    candidates = self.candidate_generator.generate_candidates()
                    pruned_candidates = self.candidate_generator.prune_candidates(candidates)

                    self.problem.print_msg(
                        f"Testing candidates (depth: {depth}, complexity: {complexity}, iteration: {iteration + 1}):\n",
                        level=1
                    )
                    func_strs = [f"{func_name}: {candidate}" for candidate, func_name in pruned_candidates]
                    candidate_functions = [candidate for candidate, _ in pruned_candidates]
                    if self.test_candidates(func_strs, candidate_functions):
                        self.problem.print_msg(f"Found satisfying candidates!", level=2)
                        for candidate, func_name in pruned_candidates:
                            self.problem.print_msg(f"{func_name}: {candidate}", level=2)
                        self.set_solution_found()
                        return

        def generate_correct_function(arg_sorts: list[SortRef]) -> tuple[callable, str]:
            args = [Var(i, sort) for i, sort in enumerate(arg_sorts)]

            def max_function(*values):
                x, y = values
                #return If(If(x >= 0, x, -x) > If(y >= 0, y, -y), If(x >= 0, x, -x), If(y >= 0, y, -y))
                return If(x >=y, x, y)

            expr = max_function(*args[:2])
            func_str = f"def max_function({', '.join(str(arg) for arg in args[:2])}):\n"
            func_str += f"    return {str(expr)}\n"
            return max_function, func_str

        args = [list(self.problem.context.z3_synth_functions.values())[0].domain(i) for i in
                range(list(self.problem.context.z3_synth_functions.values())[0].arity())]
        candidate, func_str = generate_correct_function(args)

        free_variables = [Var(i, sort) for i, sort in enumerate(args)]
        candidate_function = candidate(*free_variables)
        self.problem.print_msg(
            f"Testing known correct candidate {func_str} \n",
            level=1
        )
        test_candidates = self.test_candidates([func_str], [candidate_function])
        if test_candidates:
            self.problem.print_msg(f"Found satisfying candidates!", level=2)
            self.problem.print_msg(f"{func_str}: {candidate_function.sexpr()}", level=2)
            self.set_solution_found()
            return
        self.problem.print_msg("No satisfying candidates found.", level=2)
