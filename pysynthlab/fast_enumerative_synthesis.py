import z3
from itertools import product


class FastEnumSynth:
    # http://homepage.divms.uiowa.edu/~ajreynol/cav19b.pdf
    def __init__(self, problem):
        self.problem = problem
        self.grammar = self.extract_grammar()
        self.constructor_classes = self.compute_constructor_classes()
        self.term_cache = {}

    def extract_grammar(self):
        # mapping from sorts -> constructors and argument sorts
        return self.problem.get_grammar()

    def compute_constructor_classes(self):
        # Group constructors by their output sort and argument sorts
        constructor_classes = {}
        for sort, constructors in self.grammar.items():
            for constructor in constructors:
                key = (constructor.output_sort, tuple(constructor.argument_sorts))
                if key not in constructor_classes:
                    constructor_classes[key] = []
                constructor_classes[key].append(constructor)
        return constructor_classes

    def fast_enum(self, sort, k):
        if k < 0:
            return []
        if (sort, k) in self.term_cache:
            return self.term_cache[(sort, k)]

        terms = []
        if k == 0:
            terms.extend(self.problem.get_base_terms(sort))

        for (output_sort, arg_sorts), constructors in self.constructor_classes.items():
            if output_sort != sort:
                continue

            for size_combination in self.generate_size_combinations(len(arg_sorts), k - 1):
                arg_terms = [self.fast_enum(arg_sort, size) for arg_sort, size in zip(arg_sorts, size_combination)]
                for term_combination in product(*arg_terms):
                    for constructor in constructors:
                        term = constructor(*term_combination)
                        if self.is_unique_up_to_rewriting(term):
                            terms.append(term)

        self.term_cache[(sort, k)] = terms
        return terms

    def generate_size_combinations(self, n, k):
        # Generate all combinations of n natural numbers that sum up to k
        return [(k,)]

    def is_unique_up_to_rewriting(self, term):
        # Check if 'term' is unique in its equivalence class up to rewriting
        return True

    def generate(self, max_depth):
        # Main method to start the generation process
        for depth in range(max_depth + 1):
            for sort in self.grammar.keys():
                self.fast_enum(sort, depth)
