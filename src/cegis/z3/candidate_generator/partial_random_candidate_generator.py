import ast
import random
import time
from collections import Counter
from typing import List, Tuple, Dict, Any
from z3 import *
from src.cegis.z3.candidate_generator.candidate_generator_base import CandidateGenerator
from src.cegis.z3.synthesis_problem_z3 import SynthesisProblemZ3
from src.utilities.cancellation_token import GlobalCancellationToken


class PartialRandomCandidateGenerator(CandidateGenerator):
    """
    A 'partial' random candidate generator that uses several (optional) partial satisfaction
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

        self.metrics = problem.metrics
        self.operations = self.problem.options.synthesis_parameters.operation_costs.keys()

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

            self.metrics.update_pattern_metrics(str(candidate))

        self.metrics.candidates_generated += len(candidates)
        return candidates

    def generate_random_term(self, arg_sorts: List[z3.SortRef], depth: int, complexity: int) -> z3.ExprRef:
        """
        Generate a random term (expression) based on given constraints.

        Args:
            arg_sorts (List[z3.SortRef]): List of argument sorts for the function.
            depth (int): Maximum depth of the generated expression tree.
            complexity (int): Maximum complexity allowed for the expression.

        Returns:
            z3.ExprRef: A randomly generated Z3 expression.

        Example:
            Given arg_sorts=[z3.IntSort(), z3.IntSort()], depth=3, complexity=5,
            this method might return an expression like: If(x > y, x + 2, y - 1)
        """
        GlobalCancellationToken.check_cancellation()

        args = [z3.Var(i, sort) for i, sort in enumerate(arg_sorts)]
        constants = [z3.IntVal(i) for i in range(self.min_const, self.max_const + 1)]

        op_weights = {op: self.op_complexity(op) for op in self.operations}

        def build_term(curr_depth: int, curr_complexity: int) -> z3.ExprRef:
            GlobalCancellationToken.check_cancellation()
            if curr_depth == 0 or curr_complexity <= 0:
                return random.choice(args + constants)

            available_ops = [op for op in self.operations if curr_complexity >= self.operation_costs[op]]
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
            start_time = time.time()
            score = self.partial_satisfaction_methods[method](candidate, func_name)
            end_time = time.time()
            self.metrics.update_solver_metrics(end_time - start_time)
            self.metrics.update_partial_score(score)
            scores.append(score)
        return sum(scores) / len(scores)

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
        pruned = sorted_candidates[:top_n]
        self.metrics.candidates_pruned += len(candidates) - len(pruned)
        return pruned

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