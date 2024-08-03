from typing import List, Tuple
from z3 import *
from src.cegis.z3.candidate_generator.random_candidate_generator import RandomCandidateGenerator
from src.cegis.z3.synthesis_problem_z3 import SynthesisProblemZ3


class EnhancedRandomCandidateGenerator(RandomCandidateGenerator):
    def __init__(self, problem: SynthesisProblemZ3):
        super().__init__(problem)
        self.partial_satisfaction_methods = {
            'splitting': self.check_partial_satisfaction,
            'soft_constraints': self.check_with_soft_constraints,
            'max_smt': self.max_smt_satisfaction,
            'quantitative': self.quantitative_satisfaction,
            'unsat_core': self.unsat_core_analysis,
            'fuzzy': self.fuzzy_satisfaction
        }
        self.active_methods = set()
        self.candidate_scores = {}

    def set_partial_satisfaction_method(self, method: str, active: bool):
        if method in self.partial_satisfaction_methods:
            if active:
                self.active_methods.add(method)
            else:
                self.active_methods.discard(method)
        else:
            raise ValueError(f"Unknown partial satisfaction method: {method}")

    def evaluate_candidate(self, candidate: ExprRef, func_name: str) -> float:
        if not self.active_methods:
            return 0.0

        scores = []
        for method in self.active_methods:
            score = self.partial_satisfaction_methods[method](candidate, func_name)
            scores.append(score)

        return sum(scores) / len(scores)  # Average score across all active methods

    def check_partial_satisfaction(self, candidate: ExprRef, func_name: str) -> float:
        constraints = self.problem.context.z3_constraints
        s = Solver()
        satisfied_constraints = 0
        for constraint in constraints:
            s.push()
            constraint = self.problem.substitute_constraints(
                [constraint],
                list(self.problem.context.z3_synth_functions.values()),
                [candidate])
            s.add(constraint)
            if s.check() == sat:
                satisfied_constraints += 1
            s.pop()
        return satisfied_constraints / len(constraints)

    def check_with_soft_constraints(self, candidate: ExprRef, func_name: str) -> float:
        o = Optimize()
        for constraint in self.problem.context.z3_constraints:
            o.add_soft(constraint)
        o.add(candidate == self.problem.context.z3_synth_functions[func_name])
        if o.check() == sat:
            return o.objectives()[0].value().as_long() / len(self.problem.context.z3_constraints)
        return 0.0

    def max_smt_satisfaction(self, candidate: ExprRef, func_name: str) -> float:
        o = Optimize()
        indicators = [Bool(f'ind_{i}') for i in range(len(self.problem.context.z3_constraints))]
        for ind, constraint in zip(indicators, self.problem.context.z3_constraints):
            o.add(Implies(ind, constraint))
        o.add(candidate == self.problem.context.z3_synth_functions[func_name])
        o.maximize(Sum([If(ind, 1, 0) for ind in indicators]))
        if o.check() == sat:
            return o.model().eval(Sum([If(ind, 1, 0) for ind in indicators])).as_long() / len(
                self.problem.context.z3_constraints)
        return 0.0

    def quantitative_satisfaction(self, candidate: ExprRef, func_name: str) -> float:
        s = Solver()
        s.add(candidate == self.problem.context.z3_synth_functions[func_name])
        total_diff = 0.0
        for constraint in self.problem.context.z3_constraints:
            if is_algebraic_value(constraint):
                diff = Abs(constraint)
                s.push()
                s.add(diff >= 0)
                if s.check() == sat:
                    total_diff += s.model().eval(diff).as_fraction()
                s.pop()
            else:
                total_diff += 1
        return 1.0 / (1.0 + total_diff)

    def unsat_core_analysis(self, candidate: ExprRef, func_name: str) -> float:
        s = Solver()
        s.set(unsat_core=True)
        tracked_constraints = [Const(f'c_{i}', BoolSort()) for i in range(len(self.problem.context.z3_constraints))]
        for t, c in zip(tracked_constraints, self.problem.context.z3_constraints):
            s.assert_and_track(c, t)
        s.add(candidate == self.problem.context.z3_synth_functions[func_name])
        if s.check() == unsat:
            core = s.unsat_core()
            return (len(self.problem.context.z3_constraints) - len(core)) / len(self.problem.context.z3_constraints)
        return 1.0

    def fuzzy_satisfaction(self, candidate: ExprRef, func_name: str) -> float:
        s = Solver()
        s.add(candidate == self.problem.context.z3_synth_functions[func_name])
        satisfaction_degree = Real('satisfaction_degree')
        s.add(And(satisfaction_degree >= 0, satisfaction_degree <= 1))

        # might try and expose configurability here based on constraint type?
        for constraint in self.problem.context.z3_constraints:
            s.add(Implies(constraint, satisfaction_degree == 1))
            s.add(Implies(Not(constraint), satisfaction_degree == 0))

        if s.check() == sat:
            return s.model().eval(satisfaction_degree).as_fraction()
        return 0.0

    def generate_candidates(self) -> List[Tuple[ExprRef, str]]:
        candidates = []
        for i in range(self.problem.options.synthesis_parameters.max_candidates_at_each_depth):
            candidates.append(super().generate_candidates())

        self.candidate_scores.clear()
        for candidate, func_name in candidates:
            score = self.evaluate_candidate(candidate, func_name)
            self.candidate_scores[(candidate, func_name)] = score
        return candidates

    def prune_candidates(self, candidates: List[Tuple[ExprRef, str]]) -> List[Tuple[ExprRef, str]]:
        sorted_candidates = sorted(candidates,
                                   key=lambda x: self.candidate_scores.get((x[0], x[1]), 0),
                                   reverse=True)
        top_n = self.problem.options.synthesis_parameters.max_candidates_at_each_depth
        return sorted_candidates[:top_n]
