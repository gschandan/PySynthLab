from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
from z3 import *

from src.cegis.z3.synthesis_problem import SynthesisProblem


class SynthesisStrategy(ABC):
    def __init__(self, problem: 'SynthesisProblem'):
        self.problem = problem
        self.solution_found = False

    @abstractmethod
    def generate_candidates(self) -> List[Tuple[z3.ExprRef, str]]:
        """Generate candidate functions."""
        pass

    @abstractmethod
    def prune_candidates(self, candidates: List[Tuple[z3.ExprRef, str]]) -> List[Tuple[z3.ExprRef, str]]:
        """Prune candidates based on strategy-specific criteria."""
        pass

    def test_candidates_old(self, candidates: Dict[z3.ExprRef, str]) -> bool:

        self.problem.context.enumerator_solver.reset()
        substituted_neg_constraints = self.problem.substitute_constraints(
            self.problem.context.z3_negated_constraints,
            candidates
        )
        self.problem.context.enumerator_solver.add(substituted_neg_constraints)

        if self.problem.context.enumerator_solver.check() == z3.sat:
            model = self.problem.context.enumerator_solver.model()
            counterexamples = []
            incorrect_outputs = []
            candidate_function_exprs = []

            for func, candidate, variable_mapping in zip(candidates.items(), self.problem.context.variable_mapping_dict.values()):
                free_variables = list(variable_mapping.keys())
                counterexample = {str(free_var): model.eval(declared_var, model_completion=True).as_long()
                                  for free_var, declared_var in variable_mapping.items()}

                incorrect_output = z3.simplify(z3.substitute(candidate, [(arg, z3.IntVal(value)) for arg, value in
                                                                         zip(free_variables,
                                                                             list(counterexample.values()))]))

                self.problem.print_msg(f"Counterexample: {counterexample}", level=0)
                counterexamples.append(counterexample)
                incorrect_outputs.append(incorrect_output)
                candidate_function_expr = candidate(*free_variables) if callable(candidate) else candidate
                candidate_function_exprs.append(candidate_function_expr)

                self.problem.context.counterexamples.append((func, counterexample, incorrect_output))

            self.problem.print_msg(f"Incorrect output: {incorrect_outputs}", level=0)
            return False
        else:
            self.problem.context.verification_solver.reset()
            substituted_constraints = self.problem.substitute_constraints(
                self.problem.context.z3_constraints,
                candidates
            )
            self.problem.context.verification_solver.add(substituted_constraints)
            if self.problem.context.verification_solver.check() == z3.unsat:
                self.problem.print_msg(
                    f"Verification failed for guess. Candidates violate constraints.",level=0)
                return False
            self.problem.print_msg(f"No counterexample found! Guesses should be correct.",level=0)
            return True

    def test_candidates(self, candidates: List[Tuple[z3.ExprRef, str]]) -> bool:
        for candidate, synth_func_name in candidates:
            for stored_func_name, ce, _ in self.problem.context.counterexamples:
                if stored_func_name == synth_func_name:
                    try:
                        if not self.problem.satisfies_constraints(synth_func_name, candidate, ce):
                            self.problem.print_msg(f"Candidate {synth_func_name} failed on existing counterexample",
                                                   level=0)
                            return False
                    except Exception as e:
                        self.problem.print_msg(f"Error testing candidate {synth_func_name}: {e}", level=0)
                        return False

        new_counterexamples = self.problem.generate_counterexample(candidates)
        if new_counterexamples is not None:
            for func_name, ce in new_counterexamples.items():
                self.problem.print_msg(f"New counterexample found for {func_name}: {ce}", level=0)
                self.problem.context.counterexamples.append((func_name, ce, None))
            return False

        if not self.problem.verify_candidates([c for c, _ in candidates]):
            self.problem.print_msg(f"Verification failed for candidates. They violate constraints.", level=0)
            return False

        self.problem.print_msg(f"No counterexample found! Candidates should be correct.", level=0)
        return True

    @abstractmethod
    def execute_cegis(self) -> None:
        """Execute the CEGIS loop."""
        pass

    def set_solution_found(self) -> None:
        self.solution_found = True
