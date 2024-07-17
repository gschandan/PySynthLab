from src.cegis.z3.candidate_generators.top_down_enumerative_generator import TopDownCandidateGenerator
from src.cegis.z3.synthesis_problem import SynthesisProblem
from src.cegis.z3.synthesis_strategy import SynthesisStrategy


class RandomSearchStrategyTopDown(SynthesisStrategy):

    def __init__(self, problem: SynthesisProblem):
        super().__init__(problem)
        self.problem = problem
        self.candidate_generator = TopDownCandidateGenerator(problem)

    def execute_cegis(self) -> None:
        max_iterations = self.problem.options.synthesis_parameters.max_iterations

        for iteration in range(max_iterations):
            candidates = self.candidate_generator.generate_candidates()
            pruned_candidates = self.candidate_generator.prune_candidates(candidates)

            self.problem.print_msg(
                f"Testing candidates (iteration: {iteration + 1}):\n",
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
