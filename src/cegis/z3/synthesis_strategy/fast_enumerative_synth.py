from itertools import product
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
        max_depth = self.problem.options.synthesis_parameters.max_depth
        synth_func_names = list(self.problem.context.z3_synth_functions.keys())

        if len(synth_func_names) <= 1:
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
            return
        else:
            iteration = 0
            for depth in range(max_depth + 1):
                SynthesisProblem.logger.info(f"Depth {depth}/{max_depth}")

                all_candidates = {func_name: [] for func_name in synth_func_names}
                for candidate, func_name in self.candidate_generator.generate_candidates_at_depth(depth):
                    all_candidates[func_name].append(candidate)
                for func_name, candidates in all_candidates.items():
                    SynthesisProblem.logger.debug(f"Generated {len(candidates)} candidates for {func_name} at depth {depth}")

                if any(not candidates for candidates in all_candidates.values()):
                    SynthesisProblem.logger.warning(f"Missing candidates for some functions at depth {depth}")
                    return

                for candidate_combination in product(*(all_candidates[func_name] for func_name in synth_func_names)):
                    func_strs = synth_func_names
                    candidate_functions = list(candidate_combination)

                    SynthesisProblem.logger.info(
                        f"Testing candidates: {'; '.join([f'{func}: {cand}' for func, cand in zip(func_strs, candidate_functions)])}")

                    if self.test_candidates(func_strs, candidate_functions):
                        SynthesisProblem.logger.info(f"Found satisfying candidates!")
                        for func_name, candidate in zip(func_strs, candidate_functions):
                            SynthesisProblem.logger.info(f"{func_name}: {candidate}")
                        self.set_solution_found()
                        return
                iteration += 1
                if iteration >= max_iterations:
                    SynthesisProblem.logger.info(
                        f"No satisfying candidates found within {max_iterations} iterations.")
                    return

            SynthesisProblem.logger.info(f"No solution found up to depth {max_depth}")
