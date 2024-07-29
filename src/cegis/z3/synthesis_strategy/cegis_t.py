import random
from typing import Any, Dict, List, Optional, Tuple
from z3 import *
from z3 import ExprRef

from src.cegis.z3.synthesis_problem_z3 import SynthesisProblemZ3
from src.cegis.z3.synthesis_strategy.synthesis_strategy import SynthesisStrategy

ModelObj = ModelRef
FuncInterpObj = FuncInterp
CandidateType = Dict[str, Tuple[List[Tuple[Tuple[ExprRef, ...], ExprRef]], ExprRef]]
CounterexampleType = Dict[str, ExprRef]

# experimental
# https://www.cs.ox.ac.uk/people/alessandro.abate/publications/bcADKKP18.pdf
class CegisT(SynthesisStrategy):

    def __init__(self, problem: SynthesisProblemZ3):
        super().__init__(problem)
        self.problem = problem
        self.enumerator_solver = self.problem.context.enumerator_solver
        self.verifier_solver = self.problem.context.verification_solver
        self.theory_solver = Solver()
        self.theory_solver.set('smt.macro_finder', True)
        self.encountered_counterexamples = set()
        self.term_bank = {}

    def execute_cegis(self) -> None:
        max_depth = self.problem.options.synthesis_parameters.max_depth
        max_complexity = self.problem.options.synthesis_parameters.max_complexity
        max_candidates_per_depth = self.problem.options.synthesis_parameters.max_candidates_at_each_depth

        max_iterations = self.problem.options.synthesis_parameters.max_iterations
        for iteration in range(max_iterations):

            candidate = self.synthesize()
            if candidate is None:
                continue

            candidate = self.synthesize()
            if candidate is None:
                SynthesisProblemZ3.logger.info("No candidate found")
                continue

            counterexample = self.verify(candidate)
            if counterexample is None:
                SynthesisProblemZ3.logger.info("Solution found!")
                for func_name, expr in candidate.items():
                    SynthesisProblemZ3.logger.info(f"{func_name}: {expr}")
                self.set_solution_found()
                return

            theory_constraint = self.theory_solver_phase(candidate, counterexample)

            self.add_counterexample(counterexample)
            if theory_constraint is not None:
                self.add_theory_constraint(theory_constraint)

        SynthesisProblemZ3.logger.info("Maximum iterations reached without finding a solution.")

    def synthesize(self) -> Optional[Dict[str, ExprRef]]:
        candidates = {}
        for func_name, variable_mapping in self.problem.context.variable_mapping_dict.items():
            arg_sorts = [var.sort() for var in variable_mapping.values()]
            candidate, func_str = self.generate_random_term(arg_sorts,
                                                            self.problem.options.synthesis_parameters.max_depth,
                                                            self.problem.options.synthesis_parameters.max_complexity)
            candidates[func_name] = candidate

        if self.test_candidate(candidates):
            return candidates
        for candidate in candidates.values():
            print(f"Invalid candidate: {candidate}")
        return None

    def generate_random_term(self, arg_sorts: List[z3.SortRef], depth: int, complexity: int,
                             operations: List[str] = None) -> Tuple[z3.ExprRef, str]:
        if operations is None:
            operations = ['+', '-', '*', 'If', 'Neg']

        args = [z3.Const(f'arg{i}', sort) for i, sort in enumerate(arg_sorts)]
        num_args = len(args)
        constants = [z3.IntVal(i) for i in range(self.problem.options.synthesis_parameters.min_const,
                                                 self.problem.options.synthesis_parameters.max_const + 1)]

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
        SynthesisProblemZ3.logger.info(f"Generated expression: {generated_expression}")

        func_str = f"def arithmetic_function({', '.join(f'arg{i}' for i in range(num_args))}):\n"
        func_str += f"    return {str(generated_expression)}\n"

        return generated_expression, func_str

    def generate_condition(self, args: List[z3.ExprRef]) -> z3.BoolRef | bool:
        comparisons = ['<', '<=', '>', '>=', '==', '!=']
        left = random.choice(args)
        right = random.choice(args + [z3.IntVal(random.randint(self.problem.options.synthesis_parameters.min_const,
                                                               self.problem.options.synthesis_parameters.max_const))])
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
        # experimenting with cost of operation for biasing random choice, may make this configurable
        return {'+': 1, '-': 1, '*': 2, 'If': 3, 'Neg': 1}.get(op, 0)

    def generate_candidates(self) -> Dict[str, z3.ExprRef]:
        candidates = {}
        for func_name, variable_mapping in self.problem.context.variable_mapping_dict.items():
            candidate, func_str = self.generate_random_term([x.sort() for x in list(variable_mapping.keys())],
                                                            self.problem.options.synthesis_parameters.max_depth,
                                                            self.problem.options.synthesis_parameters.max_complexity)
            candidates[func_name] = candidate

        return candidates

    def prune_candidates(self, candidates: Dict[str, z3.ExprRef]) -> Dict[str, z3.ExprRef]:
        return candidates

    def test_candidate(self, candidate: Dict[str, ExprRef]) -> bool:
        self.enumerator_solver.push()

        for constraint in self.problem.context.z3_constraints:
            self.enumerator_solver.add(constraint)

        for ce_constraint in self.problem.context.counterexamples:
            self.enumerator_solver.add(ce_constraint)

        substitutions = [(func, candidate[func_name]) for func_name, func in
                         self.problem.context.z3_synth_functions.items()]
        substituted_constraints = self.problem.substitute_candidates(self.problem.context.z3_constraints, substitutions)
        self.enumerator_solver.add(substituted_constraints)

        result = self.enumerator_solver.check() == sat
        self.enumerator_solver.pop()
        return result

    def verify(self, candidate: Dict[str, ExprRef]) -> Optional[Dict[str, ExprRef]]:
        self.verifier_solver.push()

        substitutions = [(func, candidate[func_name]) for func_name, func in
                         self.problem.context.z3_synth_functions.items()]
        substituted_neg_constraints = self.problem.substitute_candidates(self.problem.context.z3_negated_constraints,
                                                                         substitutions)
        self.verifier_solver.add(substituted_neg_constraints)

        if self.verifier_solver.check() == sat:
            model = self.verifier_solver.model()
            counterexample = self.extract_counterexample(model)
            for func_name, func in self.problem.context.z3_synth_functions.items():
                func_args = [model.eval(arg, model_completion=True)
                             for arg in self.problem.context.variable_mapping_dict[func_name].values()]
                incorrect_output = model.eval(substitute(candidate[func_name],
                                                         [(arg, model.eval(arg, model_completion=True))
                                                          for arg in self.problem.context.variable_mapping_dict[
                                                              func_name].values()]),
                                              model_completion=True)
                counterexample[f'{func_name}_incorrect_output'] = incorrect_output

            self.verifier_solver.pop()
            return counterexample
        else:
            self.verifier_solver.pop()
            return None

    def theory_solver_phase(self, candidate: Dict[str, ExprRef], counterexample: Dict[str, ExprRef]) -> Optional[
        ExprRef]:
        self.theory_solver.push()

        generalized_candidate = self.generalize_candidate(candidate)

        for constraint in self.problem.context.z3_constraints:
            self.theory_solver.add(constraint)

        substitutions = [(func, generalized_candidate[func_name]) for func_name, func in
                         self.problem.context.z3_synth_functions.items()]
        substituted_constraints = self.problem.substitute_candidates(self.problem.context.z3_constraints, substitutions)
        self.theory_solver.add(substituted_constraints)

        if self.theory_solver.check() == sat:
            model = self.theory_solver.model()
            theory_constraint = self.extract_theory_constraint(model, generalized_candidate)
            self.theory_solver.pop()
            return theory_constraint
        else:
            self.theory_solver.pop()
            return None

    def generalize_candidate(self, candidate: Dict[str, Any]) -> Dict[str, ExprRef]:
        generalized = {}
        for func_name, expr in candidate.items():
            if isinstance(expr, ExprRef):
                generalized[func_name] = self.replace_constants_with_variables(expr)
            else:
                generalized[func_name] = expr 
        return generalized

    def replace_constants_with_variables(self, expr: ExprRef) -> ExprRef:
        if is_int_value(expr):
            return Int(f'v_{expr.as_long()}')
        elif is_const(expr):
            return expr
        elif is_var(expr):
            return expr
        elif is_app(expr):
            new_args = [self.replace_constants_with_variables(arg) for arg in expr.children()]
            return expr.decl()(*new_args)
        else:
            return expr

    def extract_theory_constraint(self, model: ModelRef, generalized_candidate: Dict[str, ExprRef]) -> ExprRef:
        constraints = []
        for func_name, expr in generalized_candidate.items():
            variables = self.get_variables(expr)
            constraints.extend([v == model[v] for v in variables if str(v).startswith('v_')])
        return And(*constraints)

    def get_variables(self, expr: ExprRef) -> List[ExprRef]:
        if is_const(expr) and expr.decl().kind() == Z3_OP_UNINTERPRETED:
            return [expr]
        elif is_expr(expr):
            return list(set(sum([self.get_variables(arg) for arg in expr.children()], [])))
        else:
            return []

    def extract_counterexample(self, model: ModelRef) -> Dict[str, ExprRef]:
        counterexample = {}
        for var_name, var in self.problem.context.z3_variables.items():
            counterexample[var_name] = model.eval(var, model_completion=True)
        return counterexample

    def add_counterexample(self, counterexample: Dict[str, ExprRef]) -> None:
        ce_tuple = tuple(sorted((str(k), str(v)) for k, v in counterexample.items()))

        if ce_tuple not in self.encountered_counterexamples:
            ce_constraint = self.formulate_counterexample_constraint(counterexample)
            SynthesisProblemZ3.logger.info(f"adding ce constraint: {ce_constraint}")
            self.problem.context.counterexamples.append(ce_constraint)
            self.encountered_counterexamples.add(ce_tuple)
        else:
            SynthesisProblemZ3.logger.info(f"Skipping duplicate counterexample: {counterexample}")

    def add_theory_constraint(self, constraint: ExprRef) -> None:
        SynthesisProblemZ3.logger.info(f"Adding theory constraint: {constraint}")
        self.problem.context.z3_constraints.append(constraint)

    def formulate_counterexample_constraint(self, counterexample: Dict[str, ExprRef]) -> ExprRef:
        constraints = []
        for func_name, func in self.problem.context.z3_synth_functions.items():
            ce_args = [counterexample[arg.__str__] for arg in
                       self.problem.context.variable_mapping_dict[func_name].values()]
            ce_condition = And(*[arg == ce_arg for arg, ce_arg in
                                 zip(self.problem.context.variable_mapping_dict[func_name].values(), ce_args)])

            incorrect_output = counterexample[f'{func_name}_incorrect_output']
            constraints.append(Implies(ce_condition, func(
                *self.problem.context.variable_mapping_dict[func_name].values()) != incorrect_output))

        return And(*constraints)
