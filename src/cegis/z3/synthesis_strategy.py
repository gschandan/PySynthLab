from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
from z3 import *
from z3 import ExprRef

from src.cegis.z3.candidate_generators.candidate_generator_base import CandidateGenerator
from src.cegis.z3.candidate_generators.random_candidate_generator import RandomCandidateGenerator
from src.cegis.z3.synthesis_problem import SynthesisProblem


class SynthesisStrategy(ABC):
    def __init__(self, problem: 'SynthesisProblem', candidate_generator: CandidateGenerator = None):
        self.problem = problem
        self.solution_found = False
        if candidate_generator is None:
            self.candidate_generator = RandomCandidateGenerator(problem)
        else:
            self.candidate_generator = candidate_generator

    def generate_candidates(self) -> list[tuple[ExprRef, str]]:
        return self.candidate_generator.generate_candidates()

    @abstractmethod
    def execute_cegis(self) -> None:
        """Execute the CEGIS loop."""
        pass

    def set_solution_found(self) -> None:
        self.solution_found = True

    def set_candidate_generator(self, generator: CandidateGenerator):
        self.candidate_generator = generator

    def test_candidates(self, func_strs: List[str], candidate_functions: List[z3.ExprRef]) -> bool:
        synth_func_names = list(self.problem.context.z3_synth_functions.keys())

        if len(func_strs) != len(synth_func_names):
            raise ValueError("Number of candidate functions doesn't match number of synthesis functions")

        for func, candidate, synth_func_name in zip(func_strs, candidate_functions, synth_func_names):
            if not self.check_counterexample(synth_func_name, candidate):
                return False

        new_counterexamples = self.generate_counterexample(list(zip(candidate_functions, synth_func_names)))
        if new_counterexamples is not None:
            for func_name, ce in new_counterexamples.items():
                self.problem.print_msg(f"New counterexample found for {func_name}: {ce}", level=0)
            return False

        if not self.verify_candidates(candidate_functions):
            self.problem.print_msg(
                f"Verification failed for guess {'; '.join(func_strs)}. Candidates violate constraints.", level=0)
            return False

        self.problem.print_msg(f"No counterexample found! Guesses should be correct: {'; '.join(func_strs)}.", level=0)
        return True

    @staticmethod
    def create_args_mapping(ce: Dict[str, Any], variable_mapping: Dict[z3.ExprRef, z3.ExprRef]) -> Dict[
        z3.ExprRef, Any]:
        reverse_mapping = {str(v): k for k, v in variable_mapping.items()}
        return {reverse_mapping[var_name]: value for var_name, value in ce.items()}

    def check_counterexample(self, func_name: str, candidate: z3.ExprRef) -> bool:
        for stored_func_name, ce in self.problem.context.counterexamples:
            if stored_func_name == func_name:
                synth_func = self.problem.context.z3_synth_functions[func_name]

                substituted_constraints = self.problem.substitute_constraints(
                    self.problem.context.z3_constraints,
                    [synth_func],
                    [candidate]
                )

                solver = z3.Solver()

                for var, value in ce.items():
                    solver.add(var == value)

                solver.add(substituted_constraints)

                if solver.check() == z3.unsat:
                    return False
        return True

    def generate_counterexample(self, candidates: List[Tuple[z3.ExprRef, str]]) -> Dict[str, Dict[str, Any]] | None:
        self.problem.context.enumerator_solver.reset()
        substituted_neg_constraints = self.problem.substitute_constraints(
            self.problem.context.z3_negated_constraints,
            list(self.problem.context.z3_synth_functions.values()),
            [candidate for candidate, _ in candidates])
        self.problem.context.enumerator_solver.add(substituted_neg_constraints)

        if self.problem.context.enumerator_solver.check() == z3.sat:
            model = self.problem.context.enumerator_solver.model()
            counterexamples = {}

            for (candidate, synth_func_name) in candidates:
                variable_mapping = self.problem.context.variable_mapping_dict[synth_func_name]
                args = list(variable_mapping.values())
                ce = {arg: model.eval(arg, model_completion=True) for arg in args}
                self.problem.print_msg(f"Counterexample for {synth_func_name}: {ce}", level=0)
                counterexamples[synth_func_name] = ce
                self.problem.context.counterexamples.append((synth_func_name, ce))

            return counterexamples
        else:
            return None

    def verify_candidates(self, candidates: List[z3.ExprRef]) -> bool:
        self.problem.context.verification_solver.reset()
        substituted_constraints = self.problem.substitute_constraints(
            self.problem.context.z3_constraints,
            list(self.problem.context.z3_synth_functions.values()),
            candidates)
        self.problem.context.verification_solver.add(substituted_constraints)

        return self.problem.context.verification_solver.check() == z3.sat
