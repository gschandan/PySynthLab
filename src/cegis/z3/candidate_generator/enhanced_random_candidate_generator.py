import ast
import random
from collections import Counter
from typing import List, Tuple, Dict, Any
from z3 import *
from src.cegis.z3.candidate_generator.candidate_generator_base import CandidateGenerator
from src.cegis.z3.synthesis_problem_z3 import SynthesisProblemZ3
from src.utilities.cancellation_token import GlobalCancellationToken


class EnhancedRandomCandidateGenerator(CandidateGenerator):
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
        for func_name in self.problem.context.z3_synth_functions.keys():
            self.adjust_weights_based_on_scores(func_name)
            self.penalize_low_scoring_patterns(func_name)

        candidates = []
        for func_name, variable_mapping in self.problem.context.variable_mapping_dict.items():
            candidate = self.generate_random_term(
                self.get_arg_sorts(func_name),
                self.problem.options.synthesis_parameters.max_depth,
                self.problem.options.synthesis_parameters.max_complexity,
            )
            candidates.append((candidate, func_name))

            score = self.evaluate_candidate(candidate, func_name)
            self.candidate_scores[(candidate, func_name)] = score
            self.score_history[func_name].append((score, str(candidate)))
        return candidates

    def generate_random_term(self, arg_sorts: List[z3.SortRef], depth: int, complexity: int,
                             operations: List[str] = None) -> z3.ExprRef:
        """
        Generate a random term (expression) based on given constraints.

        Args:
            arg_sorts (List[z3.SortRef]): List of argument sorts for the function.
            depth (int): Maximum depth of the generated expression tree.
            complexity (int): Maximum complexity allowed for the expression.
            operations (List[str], optional): List of allowed operations. Defaults to ['+', '-', '*', 'If', 'Neg'].

        Returns:
            z3.ExprRef: A randomly generated Z3 expression.

        Example:
            Given arg_sorts=[z3.IntSort(), z3.IntSort()], depth=3, complexity=5,
            this method might return an expression like: If(x > y, x + 2, y - 1)
        """
        GlobalCancellationToken.check_cancellation()
        if operations is None:
            operations = ['+', '-', '*', 'ite', 'neg']

        args = [z3.Var(i, sort) for i, sort in enumerate(arg_sorts)]
        constants = [z3.IntVal(i) for i in range(self.min_const, self.max_const + 1)]

        op_weights = {op: self.op_complexity(op) for op in operations}

        def build_term(curr_depth: int, curr_complexity: int) -> z3.ExprRef:
            GlobalCancellationToken.check_cancellation()
            if curr_depth == 0 or curr_complexity <= 0:
                return random.choice(args + constants)

            available_ops = [op for op in operations if curr_complexity >= self.operation_costs[op]]
            if not available_ops:
                return random.choice(args + constants)

            op = random.choices(available_ops, weights=[1/self.operation_costs[op] for op in available_ops])[0]
            remaining_complexity = curr_complexity - self.operation_costs[op]

            op_weights[op] *= self.problem.options.synthesis_parameters.weight_multiplier

            if op in ['+', '-']:
                left = build_term(curr_depth - 1, remaining_complexity // 2)
                right = build_term(curr_depth - 1, remaining_complexity - (remaining_complexity // 2))
                return left + right if op == '+' else left - right
            elif op == '*':
                left = random.choice(args) if args else random.choice(constants)
                right = random.choice(constants)
                return left * right
            elif op == 'ite':
                condition = self.generate_condition(args)
                true_expr = build_term(curr_depth - 1, remaining_complexity // 2)
                false_expr = build_term(curr_depth - 1, remaining_complexity - (remaining_complexity // 2))
                return z3.If(condition, true_expr, false_expr)
            elif op == 'neg':
                return -build_term(curr_depth - 1, remaining_complexity)
            self.problem.logger.error(f"Unexpected operation: {op}")
            raise ValueError(f"Unexpected operation: {op}")

        generated_expression = build_term(depth, complexity)
        self.problem.logger.info(f"Generated expression: {generated_expression}")

        return generated_expression

    def generate_condition(self, args: List[z3.ExprRef]) -> z3.BoolRef | bool:

        """
        Generate a random boolean condition using the given arguments.

        Args:
            args (List[z3.ExprRef]): List of available arguments to use in the condition.

        Returns:
            z3.BoolRef | bool: A randomly generated boolean condition.

        Example:
            Given args=[x, y], this method might return conditions like:
            x < y, x == 2, y >= 0, etc.
        """
        comparisons = ['<', '<=', '>', '>=', '==', '!=']
        left = random.choice(args)
        right = random.choice(args + [z3.IntVal(random.randint(self.min_const, self.max_const))])
        op = random.choice(comparisons)

        if op == '<':
            return left < right
        elif op == '<=':
            return left <= right
        elif op == '>':
            return left > right
        elif op == '>=':
            return left >= right
        elif op == '==':
            return left == right
        else:
            return left != right

    def analyze_candidate_structure(self, candidate_str: str) -> Dict[str, int]:
        tree = ast.parse(candidate_str)
        structure = {}
        for node in ast.walk(tree):
            node_type = type(node).__name__
            structure[node_type] = structure.get(node_type, 0) + 1
        return structure

    def adjust_weights_based_on_scores(self, func_name: str):
        high_scoring_candidates = [cand for score, cand in self.score_history[func_name] if score > 0.5]
        if not high_scoring_candidates:
            return

        structures = [self.analyze_candidate_structure(cand) for cand in high_scoring_candidates]
        common_patterns = Counter()
        for struct in structures:
            common_patterns.update(struct)

        for op in self.operation_costs:
            if 'If' in common_patterns and op == 'ite':
                self.operation_costs[op] *= 0.8
            elif 'BinOp' in common_patterns and op in ['+', '-', '*']:
                self.operation_costs[op] *= 0.9
            elif 'UnaryOp' in common_patterns and op == 'neg':
                self.operation_costs[op] *= 0.95

    def penalize_low_scoring_patterns(self, func_name: str):
        low_scoring_candidates = [cand for score, cand in self.score_history[func_name] if score < 0.2]
        if not low_scoring_candidates:
            return

        structures = [self.analyze_candidate_structure(cand) for cand in low_scoring_candidates]
        common_patterns = Counter()
        for struct in structures:
            common_patterns.update(struct)

        for op in self.operation_costs:
            if 'If' in common_patterns and op == 'ite':
                self.operation_costs[op] *= 1.2
            elif 'BinOp' in common_patterns and op in ['+', '-', '*']:
                self.operation_costs[op] *= 1.1
            elif 'UnaryOp' in common_patterns and op == 'neg':
                self.operation_costs[op] *= 1.05

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
            self.metrics.update_partial_score(score)
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

    def quantitative_satisfaction(self, candidate: ExprRef, func_name: str) -> float:
        """
        Evaluate the candidate using quantitative satisfaction.

        This method measures how close the candidate is to satisfying each constraint,
        providing a more nuanced score than binary satisfaction.

        Args:
            candidate (ExprRef): The candidate solution to evaluate.
            func_name (str): The name of the function being synthesized.

        Returns:
            float: A score between 0 and 1, where 1 indicates full satisfaction.

        Example:
            score = generator.quantitative_satisfaction(candidate, 'f')
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

        This method uses Z3's unsat core providing a score and insight into how close the candidate
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
        negated = self.problem.substitute_constraints(
            self.problem.context.z3_negated_constraints,
            [self.problem.context.z3_synth_functions[func_name]],
            [candidate])
        self.solver.assert_exprs(negated)
        if self.solver.check() == unsat:
            return 1.0

        self.solver.reset()
        self.solver.set(unsat_core=True)

        substituted_constraints = self.problem.substitute_constraints(
            self.problem.context.z3_non_conjoined_constraints,
            [self.problem.context.z3_synth_functions[func_name]],
            [candidate])
        for i, constraint in enumerate(substituted_constraints):
            self.solver.assert_and_track(Not(constraint), Bool(f'c_{i}'))

        result = self.solver.check()

        if result == unsat:
            unsat_core = self.solver.unsat_core()
            satisfied_constraints = len(substituted_constraints) - len(unsat_core)
        else:
            satisfied_constraints = 0

        score = satisfied_constraints / len(substituted_constraints)

        self.solver.set(unsat_core=False)
        return score

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