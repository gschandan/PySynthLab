from src.cegis.z3.candidate_generator.top_down_enumerative_generator import TopDownCandidateGenerator
from src.cegis.z3.synthesis_problem import SynthesisProblem
from src.cegis.z3.synthesis_strategy.synthesis_strategy import SynthesisStrategy


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

            SynthesisProblem.logger.info(f"Iteration {iteration + 1}/{max_iterations}:\n")
            func_strs = [f"{func_name}: {candidate}" for candidate, func_name in pruned_candidates]
            candidate_functions = [candidate for candidate, _ in pruned_candidates]

            if self.test_candidates(func_strs, candidate_functions):
                SynthesisProblem.logger.info(f"Found satisfying candidates!")
                for candidate, func_name in pruned_candidates:
                    SynthesisProblem.logger.info(f"{func_name}: {candidate}")
                self.set_solution_found()
                return

        SynthesisProblem.logger.info("No satisfying candidates found.")
