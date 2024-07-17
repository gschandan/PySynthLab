from src.cegis.z3.synthesis_problem import SynthesisProblem
from src.cegis.z3.synthesis_strategy import SynthesisStrategy


class RandomSearchStrategyBottomUp(SynthesisStrategy):
    def __init__(self, problem: SynthesisProblem):
        super().__init__(problem)
        self.problem = problem
        self.min_const = problem.options.synthesis_parameters.min_const
        self.max_const = problem.options.synthesis_parameters.max_const

    def execute_cegis(self) -> None:
        max_depth = self.problem.options.synthesis_parameters.max_depth
        max_complexity = self.problem.options.synthesis_parameters.max_complexity
        max_candidates_per_depth = self.problem.options.synthesis_parameters.max_candidates_at_each_depth

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
                        self.problem.print_msg(f"Found satisfying candidates!")
                        for candidate, func_name in pruned_candidates:
                            self.problem.print_msg(f"{func_name}: {candidate}")
                        self.set_solution_found()
                        return

        self.problem.print_msg("No satisfying candidates found.")
