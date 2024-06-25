from abc import ABC, abstractmethod
from typing import List, Tuple
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
