from src.cegis.z3.candidate_generator.enhanced_random_candidate_generator import EnhancedRandomCandidateGenerator
from src.cegis.z3.synthesis_problem_z3 import SynthesisProblemZ3
from src.cegis.z3.synthesis_strategy.synthesis_strategy import SynthesisStrategy


class PartialSatisfactionBottomUp(SynthesisStrategy):
    def __init__(self, problem: SynthesisProblemZ3):
        super().__init__(problem)
        self.candidate_generator = EnhancedRandomCandidateGenerator(problem)

    def execute_cegis(self) -> tuple[bool, str]:
        # TODO: add this to the config/options or make available to other strategies?
        # also probably filter out candidates if they have particularly poor scores? but poor scores may be good for generating 'stronger' counterexamples
        methods = ['splitting', 'soft_constraints', 'max_smt', 'quantitative', 'unsat_core', 'fuzzy']
        for method in methods:
            self.candidate_generator.set_partial_satisfaction_method(method, True)

        max_depth = self.problem.options.synthesis_parameters.max_depth
        max_complexity = self.problem.options.synthesis_parameters.max_complexity
        max_iterations = self.problem.options.synthesis_parameters.max_iterations
        max_candidates_per_depth = self.problem.options.synthesis_parameters.max_candidates_at_each_depth

        iteration = 0

        for depth in range(1, max_depth + 1):
            for complexity in range(1, max_complexity + 1):
                for candidate_at_depth in range(max_candidates_per_depth):
                    iteration += 1

                    process_iteration = self.process_iteration(iteration, max_iterations, depth, complexity,
                                                               candidate_at_depth, max_candidates_per_depth)
                    if process_iteration[0]:
                        return process_iteration

                    if iteration >= max_iterations:
                        self.log_final_results(max_iterations)
                        return process_iteration

    def process_iteration(self, iteration, max_iterations, depth, complexity, candidate_at_depth,max_candidates_per_depth) -> tuple[bool, str]:
        self.problem.logger.info(
            f"Iteration {iteration}/{max_iterations} max iterations, depth: {depth}, complexity: {complexity}, candidate at depth: {candidate_at_depth + 1}/{max_candidates_per_depth}):")

        candidates = self.candidate_generator.generate_candidates()
        pruned_candidates = self.candidate_generator.prune_candidates(candidates)

        func_strs = [f"{func_name}: {candidate}" for candidate, func_name in pruned_candidates]
        candidate_functions = [candidate for candidate, _ in pruned_candidates]

        if self.test_candidates(func_strs, candidate_functions):
            self.log_solution_found(pruned_candidates)
            valid_candidates = ''
            for candidate, func_name in pruned_candidates:
                valid_candidates += f"{func_name}: {candidate}\n"
            self.set_solution_found()
            return True, valid_candidates

        return False, f"No satisfying candidates found."

    def log_solution_found(self, pruned_candidates):
        self.problem.logger.info(f"Found satisfying candidates!")
        for candidate, func_name in pruned_candidates:
            self.problem.logger.info(f"{func_name}: {candidate}")
        self.set_solution_found()
        self.log_final_results()
        self.problem.logger.info("\n"+"=" * 100)

    def log_final_results(self, max_iterations=None):
        if max_iterations:
            self.problem.logger.info(f"No satisfying candidates found within {max_iterations} iterations.")
        for func_name in self.problem.context.z3_synth_functions.keys():
            self.problem.logger.info(f"Candidate Stats: {self.candidate_generator.get_score_statistics(func_name)}")
            self.problem.logger.debug(f"Final Scores (higher is better):")
            sorted_scores = sorted(self.candidate_generator.score_history[func_name],
                                   key=lambda x: x[0], reverse=True)

            joined_scores = "\n"
            for score, candidate in sorted_scores:
                joined_scores += f"{score:.4f}: {candidate}\n"

            self.problem.logger.debug(joined_scores)
