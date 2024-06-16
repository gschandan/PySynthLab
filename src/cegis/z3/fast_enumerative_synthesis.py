import z3
from itertools import product

from src.cegis.z3.synthesis_problem_z3 import SynthesisProblem


# http://homepage.divms.uiowa.edu/~ajreynol/cav19b.pdf
class FastEnumerativeSynthesis(SynthesisProblem):
    def __init__(self, problem: str, options: object = None):
        super().__init__(problem, options)
        self.grammar = self.extract_grammar()
        self.constructor_classes = self.compute_constructor_classes()
        self.term_cache = {}

    def extract_grammar(self):
        grammar = {}
        func_descriptor: z3.FuncDeclRef
        for func_name, func_descriptor in self.context.z3_synth_functions.items():
            output_sort = func_descriptor.range()
            arg_sorts = [func_descriptor.domain(i) for i in range(func_descriptor.arity())]
            if output_sort not in grammar:
                grammar[output_sort] = []
            grammar[output_sort].append((func_name, arg_sorts))
        return grammar

    def compute_constructor_classes(self):
        constructor_classes = {}
        for sort, constructors in self.grammar.items():
            for constructor in constructors:
                func_name, arg_sorts = constructor
                key = (sort, tuple(arg_sorts))
                if key not in constructor_classes:
                    constructor_classes[key] = []
                constructor_classes[key].append(func_name)
        return constructor_classes

    def fast_enum(self, sort, depth):
        if depth < 0:
            return []
        if (sort, depth) in self.term_cache:
            return self.term_cache[(sort, depth)]

        terms = []
        if depth == 0:
            terms.extend(self.get_base_terms(sort))

        for (output_sort, arg_sorts), constructors in self.constructor_classes.items():
            if output_sort != sort:
                continue

            for size_combination in self.generate_size_combinations(len(arg_sorts), depth - 1):
                arg_terms = [self.fast_enum(arg_sort, size) for arg_sort, size in zip(arg_sorts, size_combination)]
                for term_combination in product(*arg_terms):
                    for constructor in constructors:
                        term = self.construct_term(constructor, term_combination)
                        simplified_term = z3.simplify(term)
                        if self.is_unique_up_to_rewriting(simplified_term, depth):
                            terms.append(simplified_term)

        self.term_cache[(sort, depth)] = terms
        return terms

    def get_base_terms(self, sort):
        base_terms = []
        for var_name, var in self.context.z3_variables.items():
            if var.sort() == sort:
                base_terms.append(var)
        return base_terms

    def construct_term(self, constructor, term_combination):
        # need to add other operations
        func = self.context.z3_synth_functions.get(constructor)
        if func is None:
            raise ValueError(f"Unsupported constructor: {constructor}")
        return func(*term_combination)

    def generate_size_combinations(self, n, k):
        if n == 1:
            return [(k,)]
        combinations = []
        for i in range(k + 1):
            for subcombination in self.generate_size_combinations(n - 1, k - i):
                combinations.append((i,) + subcombination)
        return combinations

    def is_unique_up_to_rewriting(self, term, depth):
        simplified_term = z3.simplify(term)
        for cached_term in self.term_cache.get((term.sort(), depth), []):
            if z3.eq(simplified_term, cached_term):
                return False
        return True

    def generate(self, max_depth):
        generated_terms = {}
        for depth in range(max_depth + 1):
            for sort in self.grammar.keys():
                terms = self.fast_enum(sort, depth)
                for term in terms:
                    if self.is_unique_up_to_rewriting(term, depth):
                        result = self.test_multiple_candidates([term.sexpr()],[term])
                        if result:
                            print(f"Found satisfying term: {term}")
                            return term
                        else:
                            print(f"Term {term} does not satisfy the constraints")
                        if sort not in generated_terms:
                            generated_terms[sort] = []
                        generated_terms[sort].append(term)
                    else:
                        print(f"Non unique term {term} depth {depth} term_cache {self.term_cache}")
        return generated_terms
