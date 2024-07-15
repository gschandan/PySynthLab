from src.cegis.z3.candidate_generators.fast_enumerative_candidate_generator import FastEnumerativeSynthesisGenerator
from src.cegis.z3.synthesis_problem import SynthesisProblem
from src.cegis.z3.synthesis_strategy import SynthesisStrategy


# http://homepage.divms.uiowa.edu/~ajreynol/cav19b.pdf
class FastEnumerativeSynthesis(SynthesisStrategy):

    def __init__(self, problem: SynthesisProblem):
        super().__init__(problem)
        self.problem = problem
        self.candidate_generator = FastEnumerativeSynthesisGenerator(problem)

    def execute_cegis(self) -> None:
        for candidates in self.candidate_generator.generate_candidates():
            for candidate in candidates:
                if self.test_candidates([candidate[1]], [candidate[0]]):
                    self.problem.print_msg(f"Found satisfying candidates!", level=2)
                    for candidate, func_name in candidates:
                        self.problem.print_msg(f"{func_name}: {candidate}", level=2)
                    self.set_solution_found()
                    return

        self.problem.print_msg(f"No solution found up to depth {self.problem.options.synthesis_parameters_max_depth}",
                               level=2)
