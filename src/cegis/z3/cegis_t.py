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

            self.add_counterexample(candidate, counterexample)

    def synthesize(self) -> Optional[CandidateType]:
        self.enumerator_solver.push()
        for constraint in self.problem.context.z3_constraints:
            self.enumerator_solver.add(constraint)

        for ce in self.problem.context.counterexamples:
            self.enumerator_solver.add(self.formulate_counterexample_constraint(ce))

        if self.enumerator_solver.check() == sat:
            model = self.enumerator_solver.model()
            candidate = self.extract_candidate(model)
            self.enumerator_solver.pop()
            return candidate
        else:
            self.enumerator_solver.pop()
            return None

    def verify(self, candidate: CandidateType) -> Optional[CounterexampleType]:
        self.verifier_solver.push()
        substituted_neg_constraints = self.substitute_candidate(self.problem.context.z3_negated_constraints, candidate)
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

    def interpret_function(self, func_interp: FuncInterpObj) -> Tuple[List[Tuple[Tuple[ExprRef, ...], ExprRef]], ExprRef]:
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

    def substitute_candidate(self, constraints: ExprRef, candidate: CandidateType) -> ExprRef:
        substitutions = []
        for func_name, (entries, else_value) in candidate.items():
            func = self.problem.context.z3_synth_functions[func_name]
            args = [arg for arg in self.problem.context.variable_mapping_dict[func_name].keys()]
            body = else_value
            for entry_args, entry_value in entries:
                condition = And(*[arg == entry_arg for arg, entry_arg in zip(args, entry_args)])
                body = If(condition, entry_value, body)
            substitutions.append((func, Lambda(args, body)))

        return substitute_debug(constraints, *substitutions)

    def extract_counterexample(self, model: ModelObj) -> CounterexampleType:
        counterexample = {}
        for var_name, var in self.problem.context.z3_variables.items():
            counterexample[var_name] = model.eval(var, model_completion=True)
        return counterexample

    def add_counterexample(self, candidate: CandidateType, counterexample: CounterexampleType) -> None:
        ce_constraint = self.formulate_counterexample_constraint(candidate, counterexample)
        self.problem.context.counterexamples.append(ce_constraint)

    def formulate_counterexample_constraint(self, candidate: CandidateType,
                                            counterexample: CounterexampleType) -> ExprRef:
        constraints = []
        for func_name, (entries, else_value) in candidate.items():
            func = self.problem.context.z3_synth_functions[func_name]
            args = [self.problem.context.z3_variables[arg_name] for arg_name in
                    self.problem.context.z3_synth_function_args[func_name]]
            ce_args = [counterexample[arg_name] for arg_name in self.problem.context.z3_synth_function_args[func_name]]

            body = else_value
            for entry_args, entry_value in entries:
                condition = And(*[arg == entry_arg for arg, entry_arg in zip(args, entry_args)])
                body = If(condition, entry_value, body)

            expected_output = func(*ce_args)
            constraint = func(*args) == expected_output
            constraints.append(
                ForAll(args, Implies(And(*[arg == ce_arg for arg, ce_arg in zip(args, ce_args)]), constraint)))

        return And(*constraints)

    def print_counterexample(self, counterexample: CounterexampleType) -> None:
        print("Counterexample found:")
        for var_name, value in counterexample.items():
            print(f"  {var_name} = {value}")
