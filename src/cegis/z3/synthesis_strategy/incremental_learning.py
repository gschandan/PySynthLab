from typing import List, Tuple, Dict, Any
from z3 import z3
from src.cegis.z3.synthesis_problem import SynthesisProblem
from src.cegis.z3.synthesis_strategy.synthesis_strategy import SynthesisStrategy


class IncrementalLearningSynthesisStrategy(SynthesisStrategy):

    def __init__(self, problem: SynthesisProblem):
        super().__init__(problem)
        self.learned_constraints = z3.BoolVal(True)

    def generate_counterexample(self, candidates: List[Tuple[z3.ExprRef, str]]) -> Dict[str, Dict[str, Any]] | None:
        ce = super().generate_counterexample(candidates)
        if ce is not None:
            self.learn_from_counterexample(ce)
        return ce

    def learn_from_counterexample(self, ce: Dict[str, Dict[str, Any]]):
        for func_name, func_ce in ce.items():
            func = self.problem.context.z3_synth_functions[func_name]
            args = [z3.Const(f'{func_name}_arg_{i}', arg_sort) for i, arg_sort in enumerate(func.domain())]
            ce_constraint = z3.And([arg == value for arg, value in func_ce.items() if arg != 'output'])
            output_constraint = func(*args) != func_ce['output']
            self.learned_constraints = z3.And(self.learned_constraints, z3.Implies(ce_constraint, output_constraint))

    def verify_candidates(self, candidates: List[z3.ExprRef]) -> bool:
        self.problem.context.verification_solver.push()
        self.problem.context.verification_solver.add(self.learned_constraints)
        result = super().verify_candidates(candidates)
        self.problem.context.verification_solver.pop()
        return result

    def execute_cegis(self) -> None:
        pass
