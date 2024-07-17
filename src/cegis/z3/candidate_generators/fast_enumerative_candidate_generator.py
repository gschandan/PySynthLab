from functools import lru_cache
from typing import Dict, List, Tuple, Generator
import z3
from itertools import product, combinations_with_replacement

from src.cegis.z3.synthesis_problem import SynthesisProblem
from src.cegis.z3.candidate_generators.candidate_generator_base import CandidateGenerator


class FastEnumerativeSynthesisGenerator(CandidateGenerator):

    def __init__(self, problem: SynthesisProblem):
        super().__init__(problem)
        self.grammar = self.extract_grammar()
        self.constructor_classes = self.compute_constructor_classes()
        self.term_cache: Dict[Tuple[z3.SortRef, int], List[z3.ExprRef]] = {}
        self.candidate_cache: Dict[Tuple[str, int], List[z3.ExprRef]] = {}

    @lru_cache(maxsize=None)
    def extract_grammar(self) -> Dict[z3.SortRef, List[Tuple[str, List[z3.SortRef]]]]:
        grammar = {z3.IntSort(): [], z3.BoolSort(): []}

        for func_name, func_descriptor in self.problem.context.z3_synth_functions.items():
            output_sort = func_descriptor.range()

            if output_sort not in grammar:
                grammar[output_sort] = []

            # need to decide if to allow recursive functions or not
            # arg_sorts = [func_descriptor.domain(i) for i in range(func_descriptor.arity())]
            # grammar[output_sort].append((func_name, arg_sorts))

        grammar[z3.IntSort()].extend([
            ("Plus", [z3.IntSort(), z3.IntSort()]),
            ("Minus", [z3.IntSort(), z3.IntSort()]),
            ("Neg", [z3.IntSort()]),
            ("Ite", [z3.BoolSort(), z3.IntSort(), z3.IntSort()])
        ])
        for const in range(self.min_const, self.max_const + 1):
            grammar[z3.IntSort()].extend([
                (f"Times{const}", [z3.IntSort()]),
                (f"Const_{const}", [])
            ])

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

        return grammar

    @lru_cache(maxsize=None)
    def compute_constructor_classes(self) -> Dict[z3.SortRef, List[List[Tuple[str, List[z3.SortRef]]]]]:
        """
        Computes constructor classes for each sort in the grammar.

        :return: A dictionary mapping each sort to a list of its constructor classes.
        """
        constructor_classes = {sort: [] for sort in self.grammar}
        for sort, constructors in self.grammar.items():
            for func_name, arg_sorts in constructors:
                found = False
                for constructor_class in constructor_classes[sort]:
                    if arg_sorts == constructor_class[0][1]:
                        constructor_class.append((func_name, arg_sorts))
                        found = True
                        break
                if not found:
                    constructor_classes[sort].append([(func_name, arg_sorts)])
        return constructor_classes

    def fast_enum(self, sort: z3.SortRef, size: int) -> List[z3.ExprRef]:
        """
        Enumerates terms of the given sort and size.

        :param sort: The Z3 sort of the terms to enumerate.
        :param size: The maximum size of the terms.
        :return: A list of Z3 expressions representing the enumerated terms.
        """
        if size < 0:
            return

        cache_key = (sort, size)
        if cache_key in self.term_cache:
            yield from self.term_cache[cache_key]
            return

        terms = []
        if size == 0:
            if sort == z3.BoolSort():
                terms = [z3.BoolVal(b) for b in [True, False]]
            else:
                terms = [z3.Var(i, sort) for i in range(self.get_arity(sort))]
        else:
            for constructor_class in self.constructor_classes.get(sort, []):
                for constructor, arg_sorts in constructor_class:
                    for size_combination in self.generate_size_combinations(len(arg_sorts), size - 1):
                        arg_terms = [list(self.fast_enum(arg_sort, sub_size)) for arg_sort, sub_size in
                                     zip(arg_sorts, size_combination)]
                        for term_combination in product(*arg_terms):
                            term = self.construct_term(constructor, term_combination)
                            simplified_term = z3.simplify(term)
                            if simplified_term not in terms:
                                terms.append(simplified_term)
                                yield simplified_term

        self.term_cache[cache_key] = terms

    @lru_cache(maxsize=None)
    def construct_term(self, constructor: str, term_combination: Tuple[z3.ExprRef, ...]) -> z3.ExprRef:
        """
        Constructs a Z3 term using the given LIA constructor and arguments.

        :param constructor: The name of the LIA constructor.
        :param term_combination: A tuple of Z3 expressions representing the arguments.
        :return: The constructed Z3 term.
        """
        if constructor == 'Plus':
            return term_combination[0] + term_combination[1]
        elif constructor == 'Minus':
            return term_combination[0] - term_combination[1]
        elif constructor.startswith('Times'):
            const = int(constructor[5:])
            return const * term_combination[0]
        elif constructor == 'Neg':
            return -term_combination[0]
        elif constructor.startswith('Const_'):
            const = int(constructor[6:])
            return z3.IntVal(const)
        elif constructor in ["LE", "GE", "LT", "GT", "Eq"]:
            arg1, arg2 = term_combination
            return z3.If(
                arg1 <= arg2 if constructor == "LE" else
                arg1 >= arg2 if constructor == "GE" else
                arg1 < arg2 if constructor == "LT" else
                arg1 > arg2 if constructor == "GT" else
                arg1 == arg2,
                z3.BoolVal(True),
                z3.BoolVal(False)
            )
        elif constructor in ['And', 'Or', 'Implies', 'Xor']:
            return getattr(z3, constructor)(term_combination[0], term_combination[1])
        elif constructor == 'Not':
            return z3.Not(term_combination[0])
        elif constructor == 'Ite':
            return z3.If(term_combination[0], term_combination[1], term_combination[2])
        else:
            raise ValueError(f"Unsupported constructor: {constructor}")

    @lru_cache(maxsize=None)
    def generate_size_combinations(self, num_args: int, total_size: int) -> List[Tuple[int, ...]]:
        """
          Generates all possible combinations of argument sizes that sum up to the total size.

          :param num_args: The number of arguments.
          :param total_size: The total size of the arguments.
          :return: A list of tuples representing the size combinations.
        """
        return list(combinations_with_replacement(range(total_size + 1), num_args))

    @lru_cache(maxsize=None)
    def get_arity(self, sort: z3.SortRef) -> int:
        return max((func.arity() for func in self.problem.context.z3_synth_functions.values() if func.range() == sort),
                   default=0)

    def generate_candidates(self) -> Generator[List[Tuple[z3.ExprRef, str]], None, None]:
        max_depth = self.config.synthesis_parameters.max_depth

        for depth in range(max_depth + 1):
            for func_name, func in self.problem.context.z3_synth_functions.items():
                cache_key = (func_name, depth)
                if cache_key not in self.candidate_cache:
                    arg_sorts = self.get_arg_sorts(func_name)
                    self.candidate_cache[cache_key] = []
                    for term in self.fast_enum(func.range(), depth):
                        candidate = self.create_candidate_function(term, arg_sorts)
                        self.candidate_cache[cache_key].append(candidate)
                        yield candidate, func_name
                else:
                    yield from ((candidate, func_name) for candidate in self.candidate_cache[cache_key])

    def prune_candidates(self, candidates: List[Tuple[z3.ExprRef, str]]) -> List[Tuple[z3.ExprRef, str]]:
        pass
