from typing import List, Tuple, Dict, Any
from z3 import *
from src.cegis.z3.candidate_generator.random_candidate_generator import RandomCandidateGenerator
from src.cegis.z3.synthesis_problem_z3 import SynthesisProblemZ3
from src.utilities.cancellation_token import GlobalCancellationToken


# idea inspired by optimisation/maximisation of candidates here https://github.com/108anup/cegis/tree/main
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
        self.candidate_scores: Dict[Tuple[ExprRef, str], float] = {}
        self.score_history: Dict[str, List[Tuple[float, str]]] = {func_name: [] for func_name in
                                                                  self.problem.context.z3_synth_functions.keys()}
        self.solver = Solver()
        self.solver.set('smt.macro_finder', True)
        self.solver.set('timeout', self.problem.options.solver.timeout)
        self.solver.set('random_seed', self.problem.options.synthesis_parameters.random_seed)

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
            GlobalCancellationToken.check_cancellation()
            score = self.partial_satisfaction_methods[method](candidate, func_name)
            scores.append(score)
        return sum(scores) / len(scores)

    def check_partial_satisfaction(self, candidate: ExprRef, func_name: str) -> float:
        constraints = self.problem.context.z3_non_conjoined_constraints
        self.solver.reset()
        satisfied_constraints = 0
        for constraint in constraints:
            GlobalCancellationToken.check_cancellation()
            self.solver.push()
            substituted_constraint = self.problem.substitute_constraints(
                self.problem.negate_assertions([constraint]),
                [self.problem.context.z3_synth_functions[func_name]],
                [candidate])
            self.solver.add(substituted_constraint)
            if self.solver.check() == unsat:
                satisfied_constraints += 1
            self.solver.pop()
        return satisfied_constraints / len(constraints)

    def check_with_soft_constraints(self, candidate: ExprRef, func_name: str) -> float:
        o = Optimize()
        for constraint in self.problem.context.z3_non_conjoined_constraints:
            GlobalCancellationToken.check_cancellation()
            substituted_constraint = self.problem.substitute_constraints(
                self.problem.negate_assertions([constraint]),
                [self.problem.context.z3_synth_functions[func_name]],
                [candidate])
            o.add_soft(substituted_constraint)
        if o.check() == unsat:
            satisfied = o.model().eval(Sum([If(c, 1, 0) for c in o.assertions()]))
            return satisfied.as_long() / len(self.problem.context.z3_non_conjoined_constraints)
        return 0.0

    def max_smt_satisfaction(self, candidate: ExprRef, func_name: str) -> float:
        o = Optimize()
        indicators = [Bool(f'ind_{i}') for i in range(len(self.problem.context.z3_non_conjoined_constraints))]
        for ind, constraint in zip(indicators, self.problem.context.z3_non_conjoined_constraints):
            GlobalCancellationToken.check_cancellation()
            substituted_constraint = self.problem.substitute_constraints(
                self.problem.negate_assertions([constraint]),
                [self.problem.context.z3_synth_functions[func_name]],
                [candidate])[0]
            o.add(Implies(ind, substituted_constraint))
        objective = Sum([If(ind, 1, 0) for ind in indicators])
        o.maximize(objective)
        if o.check() == unsat:
            satisfied = o.model().eval(objective)
            return satisfied.as_long() / len(self.problem.context.z3_non_conjoined_constraints)
        return 0.0

    def quantitative_satisfaction(self, candidate: ExprRef, func_name: str) -> float:
        self.solver.reset()
        total_diff = 0.0
        for constraint in self.problem.context.z3_non_conjoined_constraints:
            GlobalCancellationToken.check_cancellation()
            substituted_constraint = self.problem.substitute_constraints(
                self.problem.negate_assertions([constraint]),
                [self.problem.context.z3_synth_functions[func_name]],
                [candidate])[0]
            if is_bool(substituted_constraint):
                self.solver.push()
                self.solver.add(substituted_constraint)
                if self.solver.check() == unsat:
                    total_diff += 0
                else:
                    total_diff += 1
                self.solver.pop()
            elif is_arith(substituted_constraint):
                diff = Abs(substituted_constraint)
                self.solver.push()
                self.solver.add(diff >= 0)
                if self.solver.check() == unsat:
                    diff_value = self.solver.model().eval(diff)
                    if is_rational_value(diff_value):
                        total_diff += diff_value.as_fraction()
                    elif is_int_value(diff_value):
                        total_diff += float(diff_value.as_long())
                    else:
                        total_diff += 1
                self.solver.pop()
            else:
                total_diff += 1
        return 1.0 / (1.0 + total_diff)

    def unsat_core_analysis(self, candidate: ExprRef, func_name: str) -> float:
        self.solver.reset()
        self.solver.set(unsat_core=True)
        tracked_constraints = [Const(f'c_{i}', BoolSort()) for i in
                               range(len(self.problem.context.z3_non_conjoined_constraints))]
        for t, c in zip(tracked_constraints, self.problem.context.z3_non_conjoined_constraints):
            GlobalCancellationToken.check_cancellation()
            substituted_constraint = self.problem.substitute_constraints(
                self.problem.negate_assertions([c]),
                [self.problem.context.z3_synth_functions[func_name]],
                [candidate])
            self.solver.assert_and_track(substituted_constraint[0], t)
        if self.solver.check() == unsat:
            core = self.solver.unsat_core()
            return (len(self.problem.context.z3_non_conjoined_constraints) - len(core)) / len(
                self.problem.context.z3_non_conjoined_constraints)
        self.solver.set(unsat_core=False)
        return 1.0

    def fuzzy_satisfaction(self, candidate: ExprRef, func_name: str) -> float:
        self.solver.reset()
        all_satisfied = True
        num_satisfied = 0

        for constraint in self.problem.context.z3_non_conjoined_constraints:
            GlobalCancellationToken.check_cancellation()
            substituted_constraint = self.problem.substitute_constraints(
                self.problem.negate_assertions([constraint]),
                [self.problem.context.z3_synth_functions[func_name]],
                [candidate])
            self.solver.push()
            self.solver.add(substituted_constraint)
            if self.solver.check() == unsat:
                num_satisfied += 1
            else:
                all_satisfied = False
            self.solver.pop()

        if all_satisfied:
            return 1.0
        else:
            return num_satisfied / len(self.problem.context.z3_non_conjoined_constraints)

    def generate_candidates(self) -> List[Tuple[ExprRef, str]]:
        candidates = super().generate_candidates()

        for candidate_func, func_name in candidates:
            score = self.evaluate_candidate(candidate_func, func_name)
            self.candidate_scores[(candidate_func, func_name)] = score
            self.score_history[func_name].append((score, str(candidate_func)))
        return candidates

    def prune_candidates(self, candidates: List[Tuple[ExprRef, str]]) -> List[Tuple[ExprRef, str]]:
        sorted_candidates = sorted(candidates, key=lambda x: self.candidate_scores.get((x[0], x[1]), 0), reverse=True)
        top_n = self.problem.options.synthesis_parameters.max_candidates_at_each_depth
        return sorted_candidates[:top_n]

    def get_score_statistics(self, func_name: str) -> Dict[str, Any]:
        scores = [score for score, _ in self.score_history[func_name]]
        if not scores:
            return {"avg": 0, "best": 0, "worst": 0, "best_candidate": None}

        best_score = max(scores)
        best_candidate = next(cand for score, cand in self.score_history[func_name] if score == best_score)

        return {
            "avg": sum(scores) / len(scores),
            "best": best_score,
            "worst": min(scores),
            "best_candidate": best_candidate
        }
