from typing import List, Tuple, Dict, Any
from z3 import *
from src.cegis.z3.candidate_generator.random_candidate_generator import RandomCandidateGenerator
from src.cegis.z3.synthesis_problem_z3 import SynthesisProblemZ3
from src.utilities.cancellation_token import GlobalCancellationToken


# idea inspired by optimisation/maximisation of candidates here https://github.com/108anup/cegis/tree/main
class EnhancedRandomCandidateGenerator(RandomCandidateGenerator):
    """
    An 'enhanced' random candidate generator that uses several (optional) partial satisfaction
    methods to evaluate and score candidate solutions for syntax-guided synthesis problems.

    This class extends the basic RandomCandidateGenerator by incorporating multiple
    strategies to assess and score how well a candidate solution satisfies the problem constraints.
    It allows for a more granular evaluation of candidates, potentially leading to faster
    convergence on a solution, or increased insight into the solution space for future learning.

    Key Features:
    1. Candidate scoring: Each generated candidate is scored using the active
       partial satisfaction methods, allowing for potentially more informed pruning and
       selection of promising candidates.

    2. Score history tracking: The class maintains a history of scores for each
       synthesized function, enabling analysis of the synthesis process over time.

    3. Candidate pruning: The generator can be extended to prune the list of candidates based on
       their scores, focusing the search on the most promising solutions.

    Usage:
        problem = SynthesisProblemZ3(problem_str, options)
        generator = EnhancedRandomCandidateGenerator(problem)
        generator.set_partial_satisfaction_method('fuzzy', True)
        generator.set_partial_satisfaction_method('quantitative', True)
        candidates = generator.generate_candidates()
        pruned_candidates = generator.prune_candidates(candidates)
        stats = generator.get_score_statistics('f')

    Attributes:
        partial_satisfaction_methods (dict): A dictionary mapping method names to their
            corresponding functions.
        active_methods (set): A set of currently active partial satisfaction methods.
        candidate_scores (dict): A dictionary storing scores for each candidate.
        score_history (dict): A dictionary storing the history of scores for each function.
        solver (Solver): A Z3 solver instance for constraint solving.

    """
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
        self.metrics = problem.metrics

    def set_partial_satisfaction_method(self, method: str, active: bool):
        """
        Activate or deactivate a partial satisfaction method.

        Args:
            method (str): The name of the method to set.
            active (bool): Whether to activate (True) or deactivate (False) the method.

        Raises:
            ValueError: If the specified method is unknown.

        Example:
            generator.set_partial_satisfaction_method('fuzzy', True)
        """
        if method in self.partial_satisfaction_methods:
            if active:
                self.active_methods.add(method)
            else:
                self.active_methods.discard(method)
        else:
            raise ValueError(f"Unknown partial satisfaction method: {method}")

    def evaluate_candidate(self, candidate: ExprRef, func_name: str) -> float:
        """
        Evaluate a candidate solution using all activated partial satisfaction methods.

        Args:
            candidate (ExprRef): The candidate solution to evaluate.
            func_name (str): The name of the function being synthesized.

        Returns:
            float: The average score across all active methods, or 0.0 if no methods are active.

        Example:
            score = generator.evaluate_candidate(candidate, 'f')
        """
        if not self.active_methods:
            return 0.0

        scores = []
        for method in self.active_methods:
            GlobalCancellationToken.check_cancellation()
            score = self.partial_satisfaction_methods[method](candidate, func_name)
            scores.append(score)
        return sum(scores) / len(scores)

    def check_partial_satisfaction(self, candidate: ExprRef, func_name: str) -> float:
        """
        Check partial satisfaction of constraints by the candidate solution.

        This method checks each constraint individually and returns the fraction satisfied.

        Args:
            candidate (ExprRef): The candidate solution to evaluate.
            func_name (str): The name of the function being synthesized.

        Returns:
            float: The fraction of constraints satisfied by the candidate.

        Example:
            score = generator.check_partial_satisfaction(candidate, 'f')
        """
        constraints = self.problem.context.z3_non_conjoined_constraints
        self.solver.reset()
        satisfied_constraints = 0
        for constraint in constraints:
            GlobalCancellationToken.check_cancellation()
            self.solver.push()
            if is_implies(constraint):
                antecedent, consequent = constraint.arg(0), constraint.arg(1)
                negated_constraint = And(antecedent, Not(consequent))
            else:
                negated_constraint = Not(constraint)
            substituted_constraint = self.problem.substitute_constraints(
                [negated_constraint],
                [self.problem.context.z3_synth_functions[func_name]],
                [candidate])
            self.solver.add(substituted_constraint)
            if self.solver.check() == unsat:
                satisfied_constraints += 1
            self.solver.pop()
        return satisfied_constraints / len(constraints)

    def check_with_soft_constraints(self, candidate: ExprRef, func_name: str) -> float:
        """
        Evaluate the candidate using soft constraints.

        This method uses Z3's Optimize solver to treat constraints as soft and
        maximize the number of satisfied constraints.

        For each constraint, we create an indicator (z3.Optimize doesn't support as_ast) which is added to the optimizer.
        If the indicator is true, the constraint must be satisfied. A function is used to counter the number of indicators
        that are true i.e. satisifed constraints via the maximised function.

        Args:
            candidate (ExprRef): The candidate solution to evaluate.
            func_name (str): The name of the function being synthesized.

        Returns:
            float: The fraction of soft constraints satisfied by the candidate.

        Example:
            score = generator.check_with_soft_constraints(candidate, 'f')
        """
        o = Optimize()
        indicators = []
        for i, constraint in enumerate(self.problem.context.z3_non_conjoined_constraints):
            GlobalCancellationToken.check_cancellation()
            indicator = Bool(f'ind_{i}')
            substituted_constraint = self.problem.substitute_constraints(
                [constraint],
                [self.problem.context.z3_synth_functions[func_name]],
                [candidate])[0]
            o.add(Implies(indicator, substituted_constraint))
            indicators.append(indicator)

        objective = Sum([If(ind, 1, 0) for ind in indicators])
        h = o.maximize(objective)

        if o.check() == sat:
            satisfied = o.lower(h)
            return satisfied.as_long() / len(self.problem.context.z3_non_conjoined_constraints)
        return 0.0

    def max_smt_satisfaction(self, candidate: ExprRef, func_name: str) -> float:
        """
        Evaluate the candidate using MaxSMT satisfaction.

        This method uses Z3's Optimize solver to maximize the number of satisfied
        hard constraints.

        Args:
            candidate (ExprRef): The candidate solution to evaluate.
            func_name (str): The name of the function being synthesized.

        Returns:
            float: The fraction of constraints satisfied by the candidate.

        Example:
            score = generator.max_smt_satisfaction(candidate, 'f')
        """
        o = Optimize()
        indicators = [Bool(f'ind_{i}') for i in range(len(self.problem.context.z3_non_conjoined_constraints))]
        for ind, constraint in zip(indicators, self.problem.context.z3_non_conjoined_constraints):
            GlobalCancellationToken.check_cancellation()
            if is_implies(constraint):
                antecedent, consequent = constraint.arg(0), constraint.arg(1)
                negated_constraint = And(antecedent, Not(consequent))
            else:
                negated_constraint = Not(constraint)
            substituted_constraint = self.problem.substitute_constraints(
                [negated_constraint],
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
        """
        Evaluate the candidate using MaxSMT satisfaction.

        This method uses Z3's Optimize solver to maximize the number of satisfied
        hard constraints.

        Args:
            candidate (ExprRef): The candidate solution to evaluate.
            func_name (str): The name of the function being synthesized.

        Returns:
            float: The fraction of constraints satisfied by the candidate.

        Example:
            score = generator.max_smt_satisfaction(candidate, 'f')
        """
        self.solver.reset()
        total_diff = 0.0
        for constraint in self.problem.context.z3_non_conjoined_constraints:
            GlobalCancellationToken.check_cancellation()
            if is_implies(constraint):
                antecedent, consequent = constraint.arg(0), constraint.arg(1)
                negated_constraint = And(antecedent, Not(consequent))
            else:
                negated_constraint = Not(constraint)
            substituted_constraint = self.problem.substitute_constraints(
                [negated_constraint],
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
        """
        Evaluate the candidate using unsat core analysis.

        This method uses Z3's unsat core feature to identify the minimal set of
        unsatisfiable constraints, providing insight into how close the candidate
        is to satisfying all constraints.

        Args:
            candidate (ExprRef): The candidate solution to evaluate.
            func_name (str): The name of the function being synthesized.

        Returns:
            float: The fraction of constraints not in the unsat core (i.e., satisfied).

        Example:
            score = generator.unsat_core_analysis(candidate, 'f')
        """
        self.solver.reset()
        self.solver.set(unsat_core=True)
        tracked_constraints = [Const(f'c_{i}', BoolSort()) for i in
                               range(len(self.problem.context.z3_non_conjoined_constraints))]
        for tracked_constraint, constraint in zip(tracked_constraints, self.problem.context.z3_non_conjoined_constraints):
            GlobalCancellationToken.check_cancellation()
            if is_implies(constraint):
                antecedent, consequent = constraint.arg(0), constraint.arg(1)
                negated_constraint = And(antecedent, Not(consequent))
            else:
                negated_constraint = Not(constraint)
            substituted_constraint = self.problem.substitute_constraints(
                negated_constraint,
                [self.problem.context.z3_synth_functions[func_name]],
                [candidate])
            self.solver.assert_and_track(substituted_constraint[0], tracked_constraint)
        if self.solver.check() == unsat:
            core = self.solver.unsat_core()
            return (len(self.problem.context.z3_non_conjoined_constraints) - len(core)) / len(
                self.problem.context.z3_non_conjoined_constraints)
        self.solver.set(unsat_core=False)
        return 1.0

    def fuzzy_satisfaction(self, candidate: ExprRef, func_name: str) -> float:
        """
        Evaluate the candidate using fuzzy satisfaction.

        This method checks each constraint individually but allows for partial
        satisfaction, providing a more gradual measure of constraint satisfaction.

        Args:
            candidate (ExprRef): The candidate solution to evaluate.
            func_name (str): The name of the function being synthesized.

        Returns:
            float: A score between 0 and 1, where 1 indicates full satisfaction.

        Example:
            score = generator.fuzzy_satisfaction(candidate, 'f')
        """
        self.solver.reset()
        all_satisfied = True
        num_satisfied = 0

        for constraint in self.problem.context.z3_non_conjoined_constraints:
            GlobalCancellationToken.check_cancellation()

            if is_implies(constraint):
                antecedent, consequent = constraint.arg(0), constraint.arg(1)
                negated_constraint = And(antecedent, Not(consequent))
            else:
                negated_constraint = Not(constraint)

            substituted_constraint = self.problem.substitute_constraints(
                [negated_constraint],
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
        """
        Generate and evaluate candidate solutions.

        This method generates candidates using the superclass method and then
        evaluates each candidate using the active partial satisfaction methods.

        Returns:
            List[Tuple[ExprRef, str]]: A list of tuples containing candidate
            expressions and their corresponding function names.

        Example:
            candidates = generator.generate_candidates()
        """
        candidates = super().generate_candidates()

        for candidate_func, func_name in candidates:
            score = self.evaluate_candidate(candidate_func, func_name)
            self.candidate_scores[(candidate_func, func_name)] = score
            self.score_history[func_name].append((score, str(candidate_func)))
        return candidates

    def prune_candidates(self, candidates: List[Tuple[ExprRef, str]]) -> List[Tuple[ExprRef, str]]:
        """
        Prune the list of candidates based on their scores.

        This method sorts the candidates by their scores in descending order and
        returns the top N candidates, where N is defined by the synthesis parameters.

        Args:
            candidates (List[Tuple[ExprRef, str]]): A list of candidate solutions
                and their corresponding function names.

        Returns:
            List[Tuple[ExprRef, str]]: A pruned list of the top N candidates.

        Example:
            pruned_candidates = generator.prune_candidates(candidates)
        """
        sorted_candidates = sorted(candidates, key=lambda x: self.candidate_scores.get((x[0], x[1]), 0), reverse=True)
        top_n = self.problem.options.synthesis_parameters.max_candidates_at_each_depth
        return sorted_candidates[:top_n]

    def get_score_statistics(self, func_name: str) -> Dict[str, Any]:
        """
        Generate and evaluate candidate solutions.

        This method generates candidates using the superclass method and then
        evaluates each candidate using the active partial satisfaction methods.

        Returns:
            List[Tuple[ExprRef, str]]: A list of tuples containing candidate
            expressions and their corresponding function names.

        Example:
            candidates = generator.generate_candidates()
        """
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
