import itertools
from functools import lru_cache
from typing import Dict, List, Tuple, Union, Set
import z3
from z3 import ExprRef

from src.cegis.z3.synthesis_problem import SynthesisProblem
from src.cegis.z3.synthesis_strategy import SynthesisStrategy


class FastEnumerativeSynthesisTopDown(SynthesisStrategy):

    def __init__(self, problem: SynthesisProblem):
        super().__init__(problem)
        self.problem = problem
        self.min_const = problem.options.min_const
        self.max_const = problem.options.max_const
        self.grammar = self.extract_grammar()
        self.term_cache = {}
        self.func_args = {}

    def extract_grammar(self) -> Dict[z3.SortRef, List[Tuple[str, List[z3.SortRef]]]]:
        grammar = {}
        for func_name, func_descriptor in self.problem.context.z3_synth_functions.items():
            output_sort = func_descriptor.range()
            if output_sort not in grammar:
                grammar[output_sort] = []

        if z3.IntSort() not in grammar:
            grammar[z3.IntSort()] = []
        grammar[z3.IntSort()].extend([
            ("Plus", [z3.IntSort(), z3.IntSort()]),
            ("Minus", [z3.IntSort(), z3.IntSort()]),
            ("Neg", [z3.IntSort()]),
            ("Ite", [z3.BoolSort(), z3.IntSort(), z3.IntSort()])
        ])
        for const in range(self.min_const, self.max_const + 1):
            grammar[z3.IntSort()].append((f"Times{const}", [z3.IntSort()]))
        for const in range(self.min_const, self.max_const + 1):
            grammar[z3.IntSort()].append((f"Const_{const}", []))

        if z3.BoolSort() not in grammar:
            grammar[z3.BoolSort()] = []
        grammar[z3.BoolSort()].extend([
            ("LE", [z3.IntSort(), z3.IntSort()]),
            ("GE", [z3.IntSort(), z3.IntSort()]),
            ("LT", [z3.IntSort(), z3.IntSort()]),
            ("GT", [z3.IntSort(), z3.IntSort()]),
            ("Eq", [z3.IntSort(), z3.IntSort()]),
            ("And", [z3.BoolSort(), z3.BoolSort()]),
            ("Or", [z3.BoolSort(), z3.BoolSort()]),
            ("Not", [z3.BoolSort()]),
            ("Implies", [z3.BoolSort(), z3.BoolSort()]),
            ("Xor", [z3.BoolSort(), z3.BoolSort()])
        ])
        print(f"grammar: {grammar}")
        return grammar

    def construct_term(self, constructor: str, term_combination: Tuple[z3.ExprRef | z3.ArithRef, ...]) -> z3.ExprRef:
        try:
            if constructor == 'Plus':
                result = term_combination[0] + term_combination[1]
            elif constructor == 'Minus':
                result = term_combination[0] - term_combination[1]
            elif constructor.startswith('Times'):
                const = int(constructor[5:])
                result = const * term_combination[0]
            elif constructor == 'Neg':
                result = -term_combination[0]
            elif constructor.startswith('Const_'):
                const = int(constructor[6:])
                return z3.IntVal(const)
            elif constructor in ["LE", "GE", "LT", "GT", "Eq"]:
                arg1, arg2 = term_combination
                result = (
                    arg1 <= arg2 if constructor == "LE" else
                    arg1 >= arg2 if constructor == "GE" else
                    arg1 < arg2 if constructor == "LT" else
                    arg1 > arg2 if constructor == "GT" else
                    arg1 == arg2
                )
            elif constructor == 'And':
                result = z3.And(term_combination[0], term_combination[1])
            elif constructor == 'Or':
                result = z3.Or(term_combination[0], term_combination[1])
            elif constructor == 'Not':
                result = z3.Not(term_combination[0])
            elif constructor == 'Implies':
                result = z3.Implies(term_combination[0], term_combination[1])
            elif constructor == 'Xor':
                result = z3.Xor(term_combination[0], term_combination[1])
            elif constructor == 'Ite':
                cond, true_branch, false_branch = term_combination
                if not z3.is_bool(cond):
                    raise ValueError(
                        f"Invalid condition type in 'Ite' constructor: {type(cond)}. Expected a Z3 BoolRef.")
                result = z3.If(cond, true_branch, false_branch)
            else:
                func = self.problem.context.z3_synth_functions.get(constructor)
                if func is None:
                    raise ValueError(f"Unsupported constructor: {constructor}")
                result = func(*term_combination)
            return self.simplify(result)
        except Exception as e:
            self.problem.print_msg(f"Error in construct_term: {e} {constructor} {term_combination}", level=2)
            raise

    def top_down_enumerate(self, max_depth: int) -> list[tuple[z3.FuncDeclRef, set[ExprRef]]]:
        solutions = []
        self.func_args = {}
        for func_name, func in self.problem.context.z3_synth_functions.items():
            self.func_args[func] = [z3.Var(i, func.domain(i)) for i in range(func.arity())]

        for _, func in self.problem.context.z3_synth_functions.items():
            self.problem.print_msg(f"Enumerating for function: {func.name()}", level=1)
            func_solutions = set()
            for depth in range(max_depth + 1):
                self.problem.print_msg(f"Exploring depth: {depth}", level=1)
                new_solutions = self.expand_parallel(func.range(), depth, func)
                func_solutions = func_solutions.union(new_solutions)
                self.problem.print_msg(f"Terms at depth {depth}: {len(new_solutions)}", level=1)
                self.problem.print_msg(f"Sample terms: {list(new_solutions)[:5]}", level=1)

            solutions.extend([(func, self.eliminate_equivalents(func_solutions))])
        return solutions

    def get_terminals(self, sort: z3.SortRef) -> List[Tuple[str, List[z3.SortRef]]]:
        return [rule for rule in self.grammar[sort] if not rule[1]]

    def can_expand(self, source: z3.SortRef, target: Union[z3.SortRef, Tuple[str, List[z3.SortRef]]]) -> bool:
        if isinstance(target, z3.SortRef):
            return source == target
        else:
            constructor, arg_sorts = target
            return source in self.grammar and all(arg_sort in self.grammar for arg_sort in arg_sorts)

    def expand_parallel(self, sort: z3.SortRef, depth: int, current_func: z3.FuncDeclRef) -> List[z3.ExprRef]:
        if depth == 0:
            return [arg for arg in self.func_args[current_func] if arg.sort() == sort] + \
                [self.construct_term(constructor, []) for constructor, arg_sorts in self.grammar[sort] if not arg_sorts]

        all_terms = []
        for constructor, arg_sorts in self.grammar[sort]:
            terms = self.expand_constructor(constructor, arg_sorts, depth, current_func)
            all_terms.extend(terms)
        return all_terms

    def expand_constructor(self, constructor: str, arg_sorts: List[z3.SortRef], depth: int,
                           current_func: z3.FuncDeclRef) -> List[z3.ExprRef]:
        terms = []
        for arg_terms in itertools.product(*[self.expand_parallel(s, depth - 1, current_func) for s in arg_sorts]):
            try:
                term = self.construct_term(constructor, arg_terms)
                if not self.semantic_prune(term):
                    terms.append(term)
            except Exception as e:
                self.problem.print_msg(f"Error constructing term: {e}", level=2)
        return terms

    def semantic_prune(self, term: z3.ExprRef) -> bool:
        if z3.is_add(term):
            if any(z3.is_int_value(arg) and arg.as_long() == 0 for arg in term.children()):
                return True
            if len(term.children()) == 2 and term.arg(0) == term.arg(1):
                return True
        elif z3.is_sub(term):
            if term.arg(0) == term.arg(1):
                return True
        elif z3.is_mul(term):
            if any(z3.is_int_value(arg) and arg.as_long() == 0 for arg in term.children()):
                return True
            if len(term.children()) == 2 and any(
                    z3.is_int_value(arg) and arg.as_long() == 1 for arg in term.children()):
                return True
        elif z3.is_and(term) or z3.is_or(term):
            if len(set(term.children())) < len(term.children()):
                return True

        return False

    @lru_cache(maxsize=None)
    def simplify(self, term):
        simplified = z3.simplify(term)
        if z3.is_app(simplified) and simplified.decl().name() in ['+', '*', 'and', 'or']:
            args = sorted(simplified.children(), key=lambda x: x.sort().name() + str(x))
            return simplified.decl()(args)
        return simplified

    def eliminate_equivalents(self, terms: Set[z3.ExprRef]) -> Set[z3.ExprRef]:
        unique_terms = set()
        for term in terms:
            simplified_term = z3.simplify(term)
            if simplified_term not in unique_terms:
                unique_terms.add(simplified_term)
        return unique_terms

    def generate_candidates(self) -> List[Tuple[z3.ExprRef, str]]:
        candidates = []
        solutions = self.top_down_enumerate(self.problem.options.max_depth)
        for func, func_solutions in solutions:
            self.problem.print_msg(f"Generated {len(func_solutions)} candidates for {func.name()}", level=1)
            for solution in func_solutions[:5]:  # Print first 5 solutions as samples
                self.problem.print_msg(f"Candidate: {solution}", level=1)
            candidates.extend([(solution, func.name()) for solution in func_solutions])
        return candidates

    def prune_candidates(self, candidates: List[Tuple[z3.ExprRef, str]]) -> List[Tuple[z3.ExprRef, str]]:
        return candidates

    def create_candidate_function(self, candidate_expr: z3.ExprRef, arg_sorts: List[z3.SortRef]) -> z3.ExprRef:
        args = [z3.Var(i, sort) for i, sort in enumerate(arg_sorts)]
        return z3.substitute(candidate_expr, [(arg, z3.Var(i, arg.sort())) for i, arg in enumerate(args)])

    def execute_cegis(self) -> None:
        max_depth = self.problem.options.max_depth
        synth_functions = list(self.problem.context.z3_synth_functions.items())

        candidates = self.generate_candidates()
        pruned_candidates = self.prune_candidates(candidates)

        candidates_per_function = []
        for func_name, func in synth_functions:
            arg_sorts = [func.domain(i) for i in range(func.arity())]
            candidates = [(self.create_candidate_function(term, arg_sorts), func_name)
                          for term, name in pruned_candidates if name == func_name]
            candidates_per_function.append(candidates)

        for combination in itertools.product(*candidates_per_function):
            pruned_combination = self.prune_candidates(list(combination))

            self.problem.print_msg(
                f"Testing candidates :\n{'; '.join([f'{func_name}: {candidate}' for candidate, func_name in pruned_combination])}",
                level=1
            )

            func_strs = [func_name for _, func_name in pruned_combination]
            candidate_functions = [candidate for candidate, _ in pruned_combination]

            if self.test_candidates(func_strs, candidate_functions):
                self.problem.print_msg(f"Found satisfying candidates!", level=2)
                for candidate, func_name in pruned_combination:
                    self.problem.print_msg(f"{func_name}: {candidate}", level=2)
                self.set_solution_found()
                return

        self.problem.print_msg(f"No solution found up to depth {max_depth}", level=2)
