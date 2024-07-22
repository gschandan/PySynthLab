import random

import z3
from typing import List, Tuple

from src.cegis.z3.synthesis_problem import SynthesisProblem


class TopDownCandidateGenerator:
    def __init__(self, problem: SynthesisProblem):
        self.problem = problem
        self.min_const = problem.options.synthesis_parameters.min_const
        self.max_const = problem.options.synthesis_parameters.max_const
        self.max_depth = problem.options.synthesis_parameters.max_depth
        self.grammar = self.define_grammar()
        self.explored_expressions: set[str] = set()

    def define_grammar(self):
        return {
            'EXPR': ['ARITH', 'IF'],
            'ARITH': ['ADD', 'SUB', 'MUL', 'VAR', 'CONST'],
            'ADD': ['(ARITH + ARITH)'],
            'SUB': ['(ARITH - ARITH)'],
            'MUL': ['(ARITH * CONST)'],
            'IF': ['If(COND, EXPR, EXPR)'],
            'COND': ['(ARITH < ARITH)', '(ARITH <= ARITH)', '(ARITH > ARITH)', '(ARITH >= ARITH)', '(ARITH == ARITH)'],
            'VAR': list(self.problem.context.variable_mapping_dict.keys()),
            'CONST': list(range(self.min_const, self.max_const + 1))
        }

    def generate_candidates(self) -> List[Tuple[z3.ExprRef, str]]:
        candidates = []
        for func_name in self.problem.context.variable_mapping_dict.keys():
            candidate = self.generate_term('EXPR', func_name)
            if candidate is not None:
                self.problem.logger.debug(f"refined_candidate: {candidate}")
                simplified_candidate = self.simplify_term(candidate) 
                self.problem.logger.debug(f"simplified_candidate: {simplified_candidate}")
                candidates.append((simplified_candidate, func_name))
        return candidates

    def generate_term(self, symbol: str, func_name: str, depth: int = 0) -> z3.ExprRef | None:
        if depth > self.max_depth:
            return None

        if symbol in self.problem.context.variable_mapping_dict[func_name]:
            return z3.Int(symbol)
        elif symbol == 'CONST':
            return z3.IntVal(random.choice(self.grammar['CONST']))

        production = random.choice(self.grammar[symbol])
        self.problem.logger.debug(f"production: {production}")

        subterms = [self.generate_term(sym, func_name, depth + 1) for sym in production.split() if sym != '+']
        self.problem.logger.debug(f"subterms: {subterms}")

        if any(subterm is None for subterm in subterms):
            return None

        if production.startswith('If'):
            return z3.If(*subterms)
        else:
            return eval(production, {'z3': z3, **{arg: subterms[i] for i, arg in enumerate(self.problem.context.variable_mapping_dict[func_name].keys())}})

    def simplify_term(self, term: z3.ExprRef) -> z3.ExprRef:
        return z3.simplify(term)

    def prune_candidates(self, candidates: List[Tuple[z3.ExprRef, str]]) -> List[Tuple[z3.ExprRef, str]]:
        return candidates
