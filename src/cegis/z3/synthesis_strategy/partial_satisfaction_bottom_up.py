from src.cegis.z3.candidate_generator.enhanced_random_candidate_generator import EnhancedRandomCandidateGenerator
from src.cegis.z3.synthesis_problem_z3 import SynthesisProblemZ3
from src.cegis.z3.synthesis_strategy.synthesis_strategy import SynthesisStrategy


class PartialSatisfactionBottomUp(SynthesisStrategy):
    def __init__(self, problem: SynthesisProblemZ3):
        super().__init__(problem)
        self.candidate_generator = EnhancedRandomCandidateGenerator(problem)

    def set_partial_satisfaction_method(self, method: str, active: bool):
        self.candidate_generator.set_partial_satisfaction_method(method, active)

    def execute_cegis(self) -> None:
        # testing - move to config/options if works
        self.set_partial_satisfaction_method('splitting', True)
        self.set_partial_satisfaction_method('soft_constraints', True)
        self.set_partial_satisfaction_method('max_smt', False)
        self.set_partial_satisfaction_method('quantitative', True)
        self.set_partial_satisfaction_method('unsat_core', False)
        self.set_partial_satisfaction_method('fuzzy', True)

        max_depth = self.problem.options.synthesis_parameters.max_depth
        max_complexity = self.problem.options.synthesis_parameters.max_complexity
        max_iterations = self.problem.options.synthesis_parameters.max_iterations
        iteration = 0

        for depth in range(1, max_depth + 1):
            for complexity in range(1, max_complexity + 1):
                candidates = self.candidate_generator.generate_candidates()
                pruned_candidates = self.candidate_generator.prune_candidates(candidates)

                self.problem.logger.info(f"Iteration {iteration + 1}/{max_iterations}max iterations, depth: {depth}, complexity: {complexity}")

                func_strs = [f"{func_name}: {candidate}" for candidate, func_name in pruned_candidates]
                candidate_functions = [candidate for candidate, _ in pruned_candidates]

                if self.test_candidates(func_strs, candidate_functions):
                    self.problem.logger.info(f"Found satisfying candidates!")
                    for candidate, func_name in pruned_candidates:
                        self.problem.logger.info(f"{func_name}: {candidate}")
                    self.set_solution_found()
                    return

                iteration += 1
                if iteration >= max_iterations:
                    self.problem.logger.info(f"No satisfying candidates found within {max_iterations} iterations.")
                    return

        self.problem.logger.info("No satisfying candidates found.")