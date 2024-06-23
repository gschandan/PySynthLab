import random
from typing import List, Tuple, Callable
from z3 import *
from src.cegis.z3.synthesis_problem import SynthesisProblem
from src.cegis.z3.synthesis_strategy import SynthesisStrategy


class RandomSearchStrategyBottomUp(SynthesisStrategy):
    """
    A bottom-up enumerative strategy - starting with base terms and building up to more complex expressions
    """
    def __init__(self, problem: SynthesisProblem):
        self.problem = problem
        self.min_const = problem.options.min_const
        self.max_const = problem.options.max_const

    def generate_arithmetic_function(self, arg_sorts: List[z3.SortRef], depth: int, complexity: int,
                                     operations: List[str] = None) -> Tuple[Callable, str]:
        """
        Generate an arithmetic function based on the given number of arguments, argument sorts, depth, complexity, and operations.

        :param arg_sorts: The list of sorts for each argument.
        :param depth: The maximum depth of the generated expression.
        :param complexity: The maximum complexity of the generated expression.
        :param operations: The list of allowed operations (default: ['+', '-', '*', 'If', 'Neg']).
        :return: A tuple containing the function implementation and its string representation.
        """
        if operations is None:
            operations = ['+', '-', '*', 'If', 'Neg']

        args = [z3.Var(i, sort) for i, sort in enumerate(arg_sorts)]
        num_args = len(args)

        def generate_expression(curr_depth: int, curr_complexity: int) -> z3.ExprRef:
            if curr_depth == 0 or curr_complexity == 0:
                return random.choice(args) if args and random.random() < 0.5 else z3.IntVal(
                    random.randint(self.min_const, self.max_const))

            expr = None
            op = random.choice(operations)
            if op == 'If' and num_args >= 2:
                condition = random.choice(
                    [args[i] < args[j] for i in range(num_args) for j in range(i + 1, num_args)] +
                    [args[i] <= args[j] for i in range(num_args) for j in range(i + 1, num_args)] +
                    [args[i] > args[j] for i in range(num_args) for j in range(i + 1, num_args)] +
                    [args[i] >= args[j] for i in range(num_args) for j in range(i + 1, num_args)] +
                    [args[i] == args[j] for i in range(num_args) for j in range(i + 1, num_args)] +
                    [args[i] != args[j] for i in range(num_args) for j in range(i + 1, num_args)]
                )
                true_expr = generate_expression(curr_depth - 1, curr_complexity - 1)
                false_expr = generate_expression(curr_depth - 1, curr_complexity - 1)
                expr = z3.If(condition, true_expr, false_expr)
            elif op == 'If' and num_args == 1:
                condition = random.choice(
                    [args[0] < z3.IntVal(random.randint(self.min_const , self.max_const)),
                     args[0] <= z3.IntVal(random.randint(self.min_const, self.max_const)),
                     args[0] > z3.IntVal(random.randint(self.min_const , self.max_const)),
                     args[0] >= z3.IntVal(random.randint(self.min_const, self.max_const)),
                     args[0] == z3.IntVal(random.randint(self.min_const, self.max_const)),
                     args[0] != z3.IntVal(random.randint(self.min_const, self.max_const))]
                )
                true_expr = generate_expression(curr_depth - 1, curr_complexity - 1)
                false_expr = generate_expression(curr_depth - 1, curr_complexity - 1)
                expr = z3.If(condition, true_expr, false_expr)
            elif op == 'Neg':
                expr = generate_expression(curr_depth - 1, curr_complexity - 1)
                if z3.is_bool(expr):
                    return z3.Not(expr)
                else:
                    return expr * -1
            elif op in ['+', '-']:
                left_expr = generate_expression(curr_depth - 1, curr_complexity - 1)
                right_expr = generate_expression(curr_depth - 1, curr_complexity - 1)
                if op == '+':
                    expr = left_expr + right_expr
                elif op == '-':
                    expr = left_expr - right_expr
            elif op == '*': 
                left_expr = random.choice(args) if args else z3.IntVal(random.randint(self.min_const, self.max_const))
                right_expr = z3.IntVal(random.randint(self.min_const, self.max_const)) 
                expr = left_expr * right_expr
            else:
                raise ValueError(f"Unsupported operation: {op}")

            return expr

        generated_expression = generate_expression(depth, complexity)
        self.problem.print_msg(f"Generated expression: {generated_expression}", level=1)
        self.problem.print_msg(f"Expression type: {type(generated_expression)}", level=1)

        def arithmetic_function(*values):
            if len(values) != num_args:
                raise ValueError("Incorrect number of values provided.")
            simplified_expr = z3.simplify(
                z3.substitute(generated_expression, [(arg, value) for arg, value in zip(args, values)]))
            return simplified_expr

        func_str = f"def arithmetic_function({', '.join(f'arg{i}' for i in range(num_args))}):\n"
        func_str += f"    return {str(generated_expression)}\n"

        return arithmetic_function, func_str

    def execute_cegis(self) -> None:
        """
        Execute the chosen counterexample-guided inductive synthesis algorithm.
        """
        max_complexity = self.problem.options.max_complexity
        max_depth = self.problem.options.max_depth
        max_candidates_to_evaluate_at_each_depth = self.problem.options.max_candidates_at_each_depth

        tested_candidates = set()

        for depth in range(1, max_depth + 1):
            for complexity in range(1, max_complexity + 1):
                guesses = []
                for _ in range(max_candidates_to_evaluate_at_each_depth):
                    candidates = []
                    func_strs = []
                    for func_name, variable_mapping in self.problem.context.variable_mapping_dict.items():
                        candidate, func_str = self.generate_arithmetic_function(
                            [x.sort() for x in list(variable_mapping.keys())], depth, complexity)
                        candidates.append(candidate)
                        func_strs.append(func_str)

                    simplified_candidates = [z3.simplify(candidate(*free_variables)) for candidate, free_variables in
                                             zip(candidates, self.problem.context.variable_mapping_dict.values())]

                    if str(simplified_candidates) not in tested_candidates:
                        tested_candidates.add(str(simplified_candidates))
                        guesses.append((candidates, func_strs, self.problem.context.variable_mapping_dict))

                for candidates, func_strs, variable_mapping_dict in guesses:
                    try:
                        candidate_expressions = []
                        for candidate, variable_mapping in zip(candidates, variable_mapping_dict.values()):
                            free_variables = list(variable_mapping.keys())
                            candidate_expr_representation = candidate(*free_variables)
                            candidate_expressions.append(candidate_expr_representation)

                        self.problem.print_msg(f"candidate_functions for substitution {candidate_expressions}", level=0)
                        self.problem.print_msg(
                            f"Testing guess (complexity: {complexity}, depth: {depth}): {'; '.join(func_strs)}",
                            level=1)
                        result = self.problem.test_candidates(func_strs, candidate_expressions)
                        self.problem.print_msg("\n", level=1)
                        if result:
                            self.problem.print_msg(f"Found satisfying candidates! {'; '.join(func_strs)}", level=2)
                            self.problem.print_msg("-" * 150, level=0)
                            for func, counterexample, incorrect_output in self.problem.context.counterexamples:
                                self.problem.print_msg(
                                    f"Candidate function: {func} Args:{counterexample} Output: {incorrect_output}",
                                    level=0)

                            self.problem.print_msg(f"Tested candidates: {tested_candidates}", level=0)
                            return
                        self.problem.print_msg("-" * 75, level=0)
                    except Exception as e:
                        self.problem.print_msg(f"Error occurred while testing candidates: {'; '.join(func_strs)}", level=0)
                        self.problem.print_msg(f"Error message: {str(e)}", level=0)
                        raise
        for func, counterexample, incorrect_output in self.problem.context.counterexamples:
            self.problem.print_msg(f"Candidate function: {func} Args:{counterexample} Output: {incorrect_output}", level=0)

        self.problem.print_msg(f"Tested candidates: {tested_candidates}", level=1)
        self.problem.print_msg("No satisfying candidates found.", level=2)
        self.problem.print_msg("-" * 150, level=0)
