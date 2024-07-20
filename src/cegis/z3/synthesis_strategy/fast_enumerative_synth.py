from src.cegis.z3.candidate_generator.fast_enumerative_candidate_generator import FastEnumerativeSynthesisGenerator
from src.cegis.z3.synthesis_problem import SynthesisProblem
from src.cegis.z3.synthesis_strategy.synthesis_strategy import SynthesisStrategy


# http://homepage.divms.uiowa.edu/~ajreynol/cav19b.pdf
class FastEnumerativeSynthesis(SynthesisStrategy):

    def __init__(self, problem: SynthesisProblem):
        super().__init__(problem)
        self.problem = problem
        self.candidate_generator = FastEnumerativeSynthesisGenerator(problem)

    def execute_cegis(self) -> None:
        max_iterations = self.problem.options.synthesis_parameters.max_iterations
        for iteration in range(max_iterations):
            SynthesisProblem.logger.info(f"Iteration {iteration + 1}/{max_iterations}")
            candidates = self.candidate_generator.generate_candidates()
            for candidate, func_name in candidates:
                SynthesisProblem.logger.info(f"Testing candidate: {func_name}: {str(candidate)}")
                if self.test_candidates([func_name], [candidate]):
                    SynthesisProblem.logger.info(f"Found satisfying candidate!")
                    SynthesisProblem.logger.info(f"{func_name}: {candidate}")
                    self.set_solution_found()
                    return

        SynthesisProblem.logger.info(f"No solution found after {max_iterations} iterations")
