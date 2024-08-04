from src.cegis.cvc5.synthesis_problem_cvc5 import SynthesisProblemCVC5
import cvc5

from src.cegis.cvc5.synthesis_strategy_base import SynthesisStrategyCVC5


# TODO refactor/abstract to allow swapping out z3 for cvc5
class RandomSearchStrategyBottomUpCVC5(SynthesisStrategyCVC5):
    def __init__(self, problem: SynthesisProblemCVC5):
        super().__init__(problem)
        self.problem = problem
        self.min_const = problem.options.synthesis_parameters.min_const
        self.max_const = problem.options.synthesis_parameters.max_const

    def execute_cegis(self) -> tuple[bool, str]:
        max_depth = self.problem.options.synthesis_parameters.max_depth
        max_complexity = self.problem.options.synthesis_parameters.max_complexity
        max_candidates_per_depth = self.problem.options.synthesis_parameters.max_candidates_at_each_depth
        max_iterations = self.problem.options.synthesis_parameters.max_iterations
        iteration = 0

        for depth in range(1, max_depth + 1):
            for complexity in range(1, max_complexity + 1):
                for candidate_at_depth in range(max_candidates_per_depth):
                    candidates = self.candidate_generator.generate_candidates()
                    pruned_candidates = self.candidate_generator.prune_candidates(candidates)
                    self.problem.logger.info(f"Iteration {iteration + 1}/{max_iterations} depth: {depth}, complexity: {complexity}, candidate at depth: {candidate_at_depth + 1}/{max_candidates_per_depth}):\n")
                    func_strs = [f"{func_name}: {candidate}" for candidate, func_name in pruned_candidates]
                    candidate_functions = [candidate for candidate, _ in pruned_candidates]
                    if self.test_candidates(func_strs, candidate_functions):
                        self.problem.logger.info(f"Found satisfying candidates!")
                        valid_candidates = ''
                        for candidate, func_name in pruned_candidates:
                            self.problem.logger.info(f"{func_name}: {candidate}")
                            valid_candidates += f"{func_name}: {candidate}\n"
                        self.set_solution_found()
                        return True, valid_candidates
                    iteration += 1
                    if iteration >= max_iterations:
                        self.problem.logger.info(f"No satisfying candidates found within {max_iterations} iterations.")
                        return False, f"No satisfying candidates found within {max_iterations} iterations."

    def test_candidates(self, func_strs: list[str], candidate_functions: list[cvc5.Term]) -> bool:
        # TODO
        pass
