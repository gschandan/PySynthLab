import random
from typing import List, Tuple, Dict

from z3 import *

from src.cegis.z3.synthesis_problem import SynthesisProblem
from src.cegis.z3.synthesis_strategy import SynthesisStrategy

# https://www.cs.ox.ac.uk/people/alessandro.abate/publications/bcADKKP18.pdf
class RandomSearchStrategyBottomUpCegisT(SynthesisStrategy):
    def __init__(self, problem: SynthesisProblem):
        super().__init__(problem)
        self.problem = problem
        self.min_const = problem.options.min_const
        self.max_const = problem.options.max_const
        self.term_bank = {}
        self.theory_solver = Solver()
        self.theory_solver.set('smt.macro_finder', True)

    def generate_random_term(self, arg_sorts: List[z3.SortRef], depth: int, complexity: int,
                             operations: List[str] = None) -> Tuple[z3.ExprRef, str]:
        if operations is None:
            operations = ['+', '-', '*', 'If', 'Neg']

        args = [z3.Var(i, sort) for i, sort in enumerate(arg_sorts)]
        num_args = len(args)
        constants = [z3.IntVal(i) for i in range(self.min_const, self.max_const + 1)]

        def build_term(curr_depth: int, curr_complexity: int) -> z3.ExprRef:
            if curr_depth == 0 or curr_complexity == 0:
                return random.choice(args + constants)

            available_ops = [op for op in operations if curr_complexity >= self.op_complexity(op)]
            if not available_ops:
                return random.choice(args + constants)

            op = random.choice(available_ops)
            remaining_complexity = curr_complexity - self.op_complexity(op)

            if op in ['+', '-']:
                left = build_term(curr_depth - 1, remaining_complexity // 2)
                right = build_term(curr_depth - 1, remaining_complexity - (remaining_complexity // 2))
                return left + right if op == '+' else left - right
            elif op == '*':
                left = random.choice(args) if args else random.choice(constants)
                right = random.choice(constants)
                return left * right
            elif op == 'If':
                condition = self.generate_condition(args)
                true_expr = build_term(curr_depth - 1, remaining_complexity // 2)
                false_expr = build_term(curr_depth - 1, remaining_complexity - (remaining_complexity // 2))
                return z3.If(condition, true_expr, false_expr)
            elif op == 'Neg':
                return -build_term(curr_depth - 1, remaining_complexity)

        generated_expression = build_term(depth, complexity)
        self.problem.print_msg(f"Generated expression: {generated_expression}", level=1)

        func_str = f"def arithmetic_function({', '.join(f'arg{i}' for i in range(num_args))}):\n"
        func_str += f"    return {str(generated_expression)}\n"

        return generated_expression, func_str

    def generate_condition(self, args: List[z3.ExprRef]) -> z3.BoolRef | bool:
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

    @staticmethod
    def op_complexity(op: str) -> int:
        return {'+': 1, '-': 1, '*': 2, 'If': 3, 'Neg': 1}.get(op, 0)

    def generate_candidates(self) -> List[Tuple[z3.ExprRef, str]]:
        candidates = []
        for func_name, variable_mapping in self.problem.context.variable_mapping_dict.items():
            candidate, func_str = self.generate_random_term([x.sort() for x in list(variable_mapping.keys())],
                                                            self.problem.options.max_depth,
                                                            self.problem.options.max_complexity)
            candidates.append((candidate, func_name))

        return candidates

    def prune_candidates(self, candidates: List[Tuple[z3.ExprRef, str]]) -> List[Tuple[z3.ExprRef, str]]:
        return candidates

    def execute_cegis(self) -> None:
        while not self.solution_found:
            candidates = self.generate_candidates()

            if self.test_candidates([c[1] for c in candidates], [c[0] for c in candidates]):
                self.set_solution_found()
                return

            self.theory_solver_phase(candidates)

    def theory_solver_phase(self, candidates: List[Tuple[ExprRef, str]]) -> None:
        for candidate, func_name in candidates:
            skeleton = self.get_skeleton(candidate)
            generalized = self.generalize_skeleton(skeleton)

            func = self.problem.context.z3_synth_functions[func_name]
            domain = [func.domain(i) for i in range(func.arity())]
            symbolic_func = Function(func_name, *domain, func.range())

            self.theory_solver.push()

            variables = list(self.problem.context.z3_variables.values())

            if self.problem.context.z3_constraints:
                substituted_constraints = self.problem.substitute_constraints(
                    self.problem.context.z3_constraints,
                    [func],
                    [symbolic_func(*[FreshInt() for _ in range(func.arity())])]
                )
                for constraint in substituted_constraints:
                    self.theory_solver.add(ForAll(variables, constraint))
            else:
                self.problem.print_msg("Warning: No constraints found.", level=1)

            self.theory_solver.add(symbolic_func == generalized)

            if self.theory_solver.check() == sat:
                model = self.theory_solver.model()
                constants = self.get_constants_from_model(model, generalized)
                final_candidate = self.instantiate_skeleton(skeleton, constants)
                if self.test_candidates([func_name], [final_candidate]):
                    self.set_solution_found()
                    self.problem.print_msg(f"Found solution for {func_name}: {final_candidate}", level=0)
                    return
            else:
                blocking_constraint = symbolic_func != skeleton
                self.problem.context.additional_constraints.append(blocking_constraint)

            self.theory_solver.pop()

    def get_skeleton(self, term: ExprRef) -> ExprRef:
        if is_int_value(term):
            return FreshInt()
        elif is_bool(term):
            return FreshBool()
        elif is_app(term):
            new_args = [self.get_skeleton(arg) for arg in term.children()]
            return term.decl()(*new_args)
        else:
            return term

    def generalize_skeleton(self, skeleton: ExprRef) -> ExprRef:
        holes = {}
        def replace_holes(expr):
            if is_var(expr) and expr.get_id() not in self.problem.context.z3_variables:
                if expr.get_id() not in holes:
                    holes[expr.get_id()] = FreshInt() if expr.sort() == IntSort() else FreshBool()
                return holes[expr.get_id()]
            elif is_app(expr):
                new_args = [replace_holes(arg) for arg in expr.children()]
                return expr.decl()(*new_args)
            else:
                return expr
        return replace_holes(skeleton)

    def get_constants_from_model(self, model: ModelRef, generalized: ExprRef) -> Dict[int, int]:
        constants = {}
        def collect_constants(expr):
            if is_var(expr) and expr.get_id() not in self.problem.context.z3_variables:
                constants[expr.get_id()] = model.eval(expr).as_long()
            elif is_app(expr):
                for arg in expr.children():
                    collect_constants(arg)
        collect_constants(generalized)
        return constants

    def instantiate_skeleton(self, skeleton: ExprRef, constants: Dict[int, int]) -> ExprRef:
        def replace_constants(expr):
            if is_var(expr) and expr.get_id() not in self.problem.context.z3_variables:
                return IntVal(constants[expr.get_id()])
            elif is_app(expr):
                new_args = [replace_constants(arg) for arg in expr.children()]
                return expr.decl()(*new_args)
            else:
                return expr
        return replace_constants(skeleton)
