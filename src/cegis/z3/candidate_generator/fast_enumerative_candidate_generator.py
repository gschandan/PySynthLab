from functools import lru_cache
from typing import Dict, List, Tuple, Generator
import z3
from itertools import product, combinations_with_replacement

from src.cegis.z3.synthesis_problem_z3 import SynthesisProblemZ3
from src.cegis.z3.candidate_generator.candidate_generator_base import CandidateGenerator


class FastEnumerativeCandidateGenerator(CandidateGenerator):
    """
    A candidate generator that uses fast enumerative synthesis to generate expressions.

    This class implements a strategy to efficiently enumerate candidate solutions
    for synthesis problems using a grammar-based approach and caching mechanisms.

    Attributes:
        grammar (Dict[z3.SortRef, List[Tuple[str, List[z3.SortRef]]]]): The extracted grammar for term generation.
        constructor_classes (Dict[z3.SortRef, List[List[Tuple[str, List[z3.SortRef]]]]]): Grouped constructors by argument types.
        term_cache (Dict[Tuple[z3.SortRef, int], List[z3.ExprRef]]): Cache for generated terms.
        candidate_cache (Dict[Tuple[str, int], List[z3.ExprRef]]): Cache for generated candidates.
    """
    def __init__(self, problem: SynthesisProblemZ3):
        super().__init__(problem)
        self.grammar = self.extract_grammar()
        self.constructor_classes = self.compute_constructor_classes()
        self.term_cache: Dict[Tuple[z3.SortRef, int], List[z3.ExprRef]] = {}
        self.candidate_cache: Dict[Tuple[str, int], List[z3.ExprRef]] = {}

    @lru_cache(maxsize=None)
    def extract_grammar(self) -> Dict[z3.SortRef, List[Tuple[str, List[z3.SortRef]]]]:
        """
        Extract the grammar from the synthesis problem.

        Returns:
            Dict[z3.SortRef, List[Tuple[str, List[z3.SortRef]]]]: The extracted grammar.
        """
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
        Compute constructor classes for each sort in the grammar.

        Returns:
            Dict[z3.SortRef, List[List[Tuple[str, List[z3.SortRef]]]]]: A dictionary mapping each sort to a list of its constructor classes.
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
        Enumerate terms of the given sort and size.

        Args:
            sort (z3.SortRef): The Z3 sort of the terms to enumerate.
            size (int): The maximum size of the terms.

        Yields:
            z3.ExprRef: Enumerated terms of the specified sort and size.
        """
        SynthesisProblemZ3.logger.debug(f"Entering fast_enum with sort {sort} and size {size}")
        if size < 0:
            return

        cache_key = (sort, size)
        if cache_key in self.term_cache:
            SynthesisProblemZ3.logger.debug(f"Returning {len(self.term_cache[cache_key])} cached terms for {cache_key}")
            yield from self.term_cache[cache_key]
            return

        terms = []
        if size == 0:
            SynthesisProblemZ3.logger.debug(f"Generating terms for size 0 and sort {sort}")
            if sort == z3.BoolSort():
                terms = [z3.BoolVal(b) for b in [True, False]]
            else:
                terms = [z3.Var(i, sort) for i in range(self.get_arity(sort))]
            for term in terms:
                yield term
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
        SynthesisProblemZ3.logger.debug(f"Generated {len(terms)} terms for {cache_key}")
        if not terms:
            SynthesisProblemZ3.logger.warning(f"No terms generated for {cache_key}")
        self.term_cache[cache_key] = terms

    @lru_cache(maxsize=None)
    def construct_term(self, constructor: str, term_combination: Tuple[z3.ExprRef, ...]) -> z3.ExprRef:
        """
        Construct a Z3 term using the given LIA constructor and arguments.

        Args:
            constructor (str): The name of the LIA constructor.
            term_combination (Tuple[z3.ExprRef, ...]): A tuple of Z3 expressions representing the arguments.

        Returns:
            z3.ExprRef: The constructed Z3 term.

        Raises:
            ValueError: If an unsupported constructor is provided.
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
        Generate all possible combinations of argument sizes that sum up to the total size.

        Args:
            num_args (int): The number of arguments.
            total_size (int): The total size of the arguments.

        Returns:
            List[Tuple[int, ...]]: A list of tuples representing the size combinations.
        """
        return list(combinations_with_replacement(range(total_size + 1), num_args))

    @lru_cache(maxsize=None)
    def get_arity(self, sort: z3.SortRef) -> int:
        """
        Get the maximum arity of functions with the given sort as their range.

        Args:
            sort (z3.SortRef): The sort to check.

        Returns:
            int: The maximum arity found, or 0 if no functions have this sort as their range.
        """
        return max((func.arity() for func in self.problem.context.z3_synth_functions.values() if func.range() == sort),
                   default=0)

    def generate_candidates(self) -> Generator[List[Tuple[z3.ExprRef, str]], None, None]:
        """
        Generate candidates for all synthesis functions up to the maximum depth.

        Yields:
            List[Tuple[z3.ExprRef, str]]: A list of tuples, each containing a candidate expression and the function name it's for.
        """
        max_depth = SynthesisProblemZ3.options.synthesis_parameters.max_depth

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

    def generate_candidates_at_depth(self, depth: int) -> Generator[Tuple[z3.ExprRef, str], None, None]:
        """
        Generate candidates for all synthesis functions at a specific depth.

        Args:
            depth (int): The depth at which to generate candidates.

        Yields:
            Tuple[z3.ExprRef, str]: A tuple containing a candidate expression and the function name it's for.
        """
        for func_name, func in self.problem.context.z3_synth_functions.items():
            cache_key = (func_name, depth)
            if cache_key not in self.candidate_cache:
                arg_sorts = [func.domain(i) for i in range(func.arity())]
                self.candidate_cache[cache_key] = []
                terms = list(self.fast_enum(func.range(), depth))
                SynthesisProblemZ3.logger.debug(f"Generated {len(terms)} terms for {func_name} at depth {depth}")
                for term in terms:
                    candidate = self.create_candidate_function(term, arg_sorts)
                    self.candidate_cache[cache_key].append(candidate)

            for candidate in self.candidate_cache[cache_key]:
                yield candidate, func_name

        for func_name, candidates in self.candidate_cache.items():
            if not candidates:
                SynthesisProblemZ3.logger.warning(f"No candidates generated for {func_name} at depth {depth}")

    def prune_candidates(self, candidates: List[Tuple[z3.ExprRef, str]]) -> List[Tuple[z3.ExprRef, str]]:
        """
        Prune the list of candidate expressions.

        This method is a placeholder for potential future pruning strategies.
        Currently, it doesn't perform any pruning.

        Args:
            candidates (List[Tuple[z3.ExprRef, str]]): The list of candidate expressions to prune.

        Returns:
            List[Tuple[z3.ExprRef, str]]: The pruned list of candidates (currently unchanged).
        """
        return candidates
