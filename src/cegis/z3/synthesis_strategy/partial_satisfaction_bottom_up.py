import time
from src.cegis.z3.candidate_generator.partial_random_candidate_generator import PartialRandomCandidateGenerator
from src.cegis.z3.synthesis_problem_z3 import SynthesisProblemZ3
from src.cegis.z3.synthesis_strategy.synthesis_strategy import SynthesisStrategy


class PartialSatisfactionBottomUp(SynthesisStrategy):
    def __init__(self, problem: SynthesisProblemZ3):
        super().__init__(problem)
        self.candidate_generator = PartialRandomCandidateGenerator(problem)
        self.start_time = None

    def execute_cegis(self) -> tuple[bool, str]:
        methods = ['splitting', 'quantitative', 'unsat_core', 'fuzzy']
        for method in methods:
            self.candidate_generator.set_partial_satisfaction_method(method, True)

        max_depth = self.problem.options.synthesis_parameters.max_depth
        max_complexity = self.problem.options.synthesis_parameters.max_complexity
        max_iterations = self.problem.options.synthesis_parameters.max_iterations
        max_candidates_per_depth = self.problem.options.synthesis_parameters.max_candidates_at_each_depth

        iteration = 0
        self.start_time = time.time()

        total_space = sum(max_candidates_per_depth * complexity for complexity in range(1, max_complexity + 1)) * max_depth
        self.metrics.total_space = total_space

        for depth in range(1, max_depth + 1):
            for complexity in range(1, max_complexity + 1):
                for candidate_at_depth in range(max_candidates_per_depth):
                    self.metrics.iterations += 1
                    iteration += 1

                    process_iteration = self.process_iteration(iteration, max_iterations, depth, complexity,
                                                               candidate_at_depth, max_candidates_per_depth)
                    if process_iteration[0]:
                        return process_iteration

                    if iteration >= max_iterations:
                        self.log_final_results(max_iterations)
                        return process_iteration

    def process_iteration(self, iteration, max_iterations, depth, complexity, candidate_at_depth, max_candidates_per_depth) -> tuple[bool, str]:
        self.problem.logger.info(
            f"Iteration {iteration}/{max_iterations} max iterations, depth: {depth}, complexity: {complexity}, candidate at depth: {candidate_at_depth + 1}/{max_candidates_per_depth}):")

        candidates = self.candidate_generator.generate_candidates()
        self.metrics.candidates_generated += len(candidates)

        for candidate, _ in candidates:
            self.metrics.update_pattern_metrics(str(candidate))

        pruned_candidates = self.candidate_generator.prune_candidates(candidates)
        self.metrics.candidates_pruned += len(candidates) - len(pruned_candidates)

        func_strs = [f"{func_name}: {candidate}" for candidate, func_name in pruned_candidates]
        candidate_functions = [candidate for candidate, _ in pruned_candidates]

        solver_start_time = time.time()
        test_result = self.test_candidates(func_strs, candidate_functions)
        solver_time = time.time() - solver_start_time
        self.metrics.update_solver_metrics(solver_time)

        if test_result:
            self.metrics.time_spent = time.time() - self.start_time
            self.metrics.solution_found = True
            self.metrics.solution_height = depth
            self.metrics.solution_complexity = complexity
            self.log_solution_found(pruned_candidates)
            valid_candidates = ''
            for candidate, func_name in pruned_candidates:
                valid_candidates += f"{func_name}: {candidate}\n"
            self.set_solution_found()
            return True, valid_candidates

        self.metrics.time_spent = time.time() - self.start_time
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