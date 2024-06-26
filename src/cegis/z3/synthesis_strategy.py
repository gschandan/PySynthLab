from abc import ABC, abstractmethod
from random import randrange
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

    @abstractmethod
    def execute_cegis(self) -> None:
        """Execute the CEGIS loop."""
        pass

    def set_solution_found(self) -> None:
        self.solution_found = True

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

    def check_counterexample(self, func_name: str, candidate: z3.ExprRef) -> bool:
        if not any(ce[0] == func_name for ce in self.problem.context.counterexamples):
            return True

        variable_mapping = self.problem.context.variable_mapping_dict[func_name]
        args = list(variable_mapping.values())
        candidate_expr = z3.substitute_vars(candidate, *args)

        for stored_func_name, ce, _ in self.problem.context.counterexamples:
            if stored_func_name == func_name:
                substituted_expr = z3.substitute(candidate_expr, [
                    (arg, z3.IntVal(ce[str(arg)])) for arg in args
                ])
                result = z3.simplify(substituted_expr)
                if not self.satisfies_constraints(func_name, candidate_expr, result):
                    return False

        return True

    def satisfies_constraints(self, func_name: str, candidate: z3.ExprRef, result: z3.ExprRef) -> bool:
        solver = z3.Solver()
        substituted_constraints = self.problem.substitute_constraints(
            self.problem.context.z3_constraints,
            [self.problem.context.z3_synth_functions[func_name]],
            [candidate])
        solver.add(substituted_constraints)
        solver.add(self.problem.context.z3_synth_functions[func_name](
            *self.problem.context.variable_mapping_dict[func_name].values()) == result)
        return solver.check() == z3.sat

    def verify_candidates(self, candidates: List[z3.ExprRef]) -> bool:
        self.problem.context.verification_solver.reset()
        if self.problem.options.randomise_each_iteration:
            self.problem.context.verification_solver.set('random_seed', randrange(1, 4294967295))

        substituted_constraints = self.problem.substitute_constraints(
            self.problem.context.z3_constraints,
            list(self.problem.context.z3_synth_functions.values()),
            candidates)
        self.problem.context.verification_solver.add(substituted_constraints)

        return self.problem.context.verification_solver.check() == z3.sat

    def generate_counterexample(self, candidates: List[Tuple[z3.ExprRef, str]]) -> Dict[str, Dict[str, int]] | None:
        self.problem.context.enumerator_solver.reset()
        if self.problem.options.randomise_each_iteration:
            self.problem.context.enumerator_solver.set('random_seed', randrange(1, 4294967295))

        substituted_neg_constraints = self.problem.substitute_constraints(
            self.problem.context.z3_negated_constraints,
            list(self.problem.context.z3_synth_functions.values()),
            [candidate for candidate, _ in candidates])
        self.problem.context.enumerator_solver.add(substituted_neg_constraints)

        counterexamples = {}

        for (candidate, synth_func_name) in candidates:
            variable_mapping = self.problem.context.variable_mapping_dict[synth_func_name]
            func = self.problem.context.z3_synth_functions[synth_func_name]
            args = list(variable_mapping.values())
            candidate_expr = z3.substitute_vars(candidate, *args)
            difference_constraint = candidate_expr != func(*args)

            self.problem.context.enumerator_solver.push()
            self.problem.context.enumerator_solver.add(difference_constraint)

            if self.problem.context.enumerator_solver.check() == z3.sat:
                model = self.problem.context.enumerator_solver.model()
                counterexample = {str(arg): model.eval(arg, model_completion=True).as_long() for arg in args}
                incorrect_output = z3.simplify(
                    z3.substitute(candidate_expr, [(arg, z3.IntVal(counterexample[str(arg)])) for arg in args]))
                self.problem.print_msg(f"Counterexample for {synth_func_name}: {counterexample}", level=0)
                counterexamples[synth_func_name] = counterexample
                self.problem.print_msg(f"Incorrect output for {synth_func_name}: {incorrect_output}", level=0)
                self.problem.context.counterexamples.append((synth_func_name, counterexample, incorrect_output))

            self.problem.context.enumerator_solver.pop()

        return counterexamples if counterexamples else None
