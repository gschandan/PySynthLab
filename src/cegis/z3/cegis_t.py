import random
from typing import Any, Dict, List, Optional, Tuple, Union
from z3 import *
from z3 import ExprRef

from src.cegis.z3.synthesis_problem import SynthesisProblem
from src.cegis.z3.synthesis_strategy import SynthesisStrategy

ModelObj = ModelRef
FuncInterpObj = FuncInterp
CandidateType = Dict[str, Tuple[List[Tuple[Tuple[ExprRef, ...], ExprRef]], ExprRef]]
CounterexampleType = Dict[str, ExprRef]


# https://www.cs.ox.ac.uk/people/alessandro.abate/publications/bcADKKP18.pdf
class CegisT(SynthesisStrategy):

    def __init__(self, problem: SynthesisProblem):
        super().__init__(problem)
        self.problem = problem
        self.enumerator_solver = self.problem.context.enumerator_solver
        self.verifier_solver = self.problem.context.verification_solver
        self.encountered_counterexamples = set()

    def generate_candidates(self) -> List[Tuple[z3.ExprRef, str]]:
        pass

    def test_candidates(self, func_strs: List[str], candidate_functions: List[z3.ExprRef]) -> bool:
        pass

    def prune_candidates(self, candidates: List[Tuple[z3.ExprRef, str]]) -> List[Tuple[z3.ExprRef, str]]:
        pass

    def execute_cegis(self) -> None:
        while True:
            candidate = self.synthesize()
            if candidate is None:
                print("No solution found")
                return

            counterexample = self.verify(candidate)
            if counterexample is None:
                print("Solution found:", candidate)
                return

            self.add_counterexample(counterexample)

    def synthesize(self) -> Optional[CandidateType]:
        self.enumerator_solver.push()
        for constraint in self.problem.context.z3_constraints:
            self.enumerator_solver.add(constraint)

        for ce_constraint in self.problem.context.counterexamples:
            self.enumerator_solver.add(ce_constraint)

        if self.enumerator_solver.check() == sat:
            model = self.enumerator_solver.model()
            candidate = self.extract_candidate(model)
            self.enumerator_solver.pop()
            print(f"Candidate: {candidate}")
            return candidate
        else:
            self.enumerator_solver.pop()
            return None

    def verify(self, candidate: CandidateType) -> Optional[CounterexampleType]:
        self.verifier_solver.push()

        substitutions = []
        substituted_neg_constraints = []
        for func_name, (entries, else_value) in candidate.items():
            func = self.problem.context.z3_synth_functions[func_name]
            args = [Var(i, func.domain(i)) for i in range(func.arity())]
            body = else_value
            for entry_args, entry_value in reversed(entries):
                condition = And(*[arg == entry_arg for arg, entry_arg in zip(args, entry_args)])
                body = If(condition, entry_value, body)

            def func_appl(*values):
                return body

            func_app = func_appl(*args)
            substitutions.append((func, func_app))

            substituted_neg_constraints.append(self.problem.substitute_constraints(
                self.problem.context.z3_negated_constraints,
                [self.problem.context.z3_synth_functions[func_name]],
                [func_appl(*args)]

            ))

        self.verifier_solver.add(substituted_neg_constraints)

        if self.verifier_solver.check() == sat:
            model = self.verifier_solver.model()
            counterexample = self.extract_counterexample(model)
            self.verifier_solver.pop()
            return counterexample
        else:
            self.verifier_solver.pop()
            return None

    def extract_candidate(self, model: ModelObj) -> CandidateType:
        candidate: CandidateType = {}
        for func_name, func in self.problem.context.z3_synth_functions.items():
            func_interp: FuncInterpObj = model[func]
            if func_interp is None:
                raise ValueError(f"No interpretation found for function {func_name}")

            candidate[func_name] = self.interpret_function(func_interp)
        return candidate

    def interpret_function(self, func_interp: FuncInterpObj) -> Tuple[
        List[Tuple[Tuple[ExprRef, ...], ExprRef]], ExprRef]:
        entries: List[Tuple[Tuple[ExprRef, ...], ExprRef]] = []
        for i in range(func_interp.num_entries()):
            entry: FuncEntry = func_interp.entry(i)
            args: Tuple[ExprRef, ...] = tuple(entry.arg_value(j) for j in range(entry.num_args()))
            value: ExprRef = entry.value()
            entries.append((args, value))

        else_value: ExprRef = func_interp.else_value()
        if else_value is None:
            raise ValueError("Function interpretation has no else value")

        return entries, else_value

    def extract_counterexample(self, model: ModelObj) -> CounterexampleType:
        counterexample = {}
        for var_name, var in self.problem.context.z3_variables.items():
            counterexample[var_name] = model.eval(var, model_completion=True)
        return counterexample

    def add_counterexample(self, counterexample: CounterexampleType) -> None:
        ce_tuple = tuple(sorted((str(k), str(v)) for k, v in counterexample.items()))

        if ce_tuple not in self.encountered_counterexamples:
            ce_constraint = self.formulate_counterexample_constraint(counterexample)
            print(f"adding ce constraint: {ce_constraint}")
            self.problem.context.counterexamples.append(ce_constraint)
            self.encountered_counterexamples.add(ce_tuple)
        else:
            print(f"Skipping duplicate counterexample: {counterexample}")

    def formulate_counterexample_constraint(self, counterexample: CounterexampleType) -> ExprRef:
        constraints = []
        for func_name, func in self.problem.context.z3_synth_functions.items():
            args = [Const(f'{func_name}_arg_{i}', func.domain(i)) for i in range(func.arity())]
            ce_args = [counterexample[arg.__str__()]
                       for arg in self.problem.context.variable_mapping_dict[func_name].values()]

            expected_output = func(*ce_args)
            constraint = func(*args) == expected_output
            ce_condition = And(*[arg == ce_arg for arg, ce_arg in zip(args, ce_args)])
            constraints.append(Implies(ce_condition, constraint))

        return And(*constraints)
