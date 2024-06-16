from typing import Dict, List, Tuple
import z3
from itertools import product
from src.cegis.z3.synthesis_problem_z3 import SynthesisProblem


# http://homepage.divms.uiowa.edu/~ajreynol/cav19b.pdf
class FastEnumerativeSynthesis(SynthesisProblem):
    MIN_CONST = -2
    MAX_CONST = 2

    def __init__(self, problem: str, options: object = None):
        super().__init__(problem, options)
        self.grammar = self.extract_grammar()
        self.constructor_classes = self.compute_constructor_classes()
        self.term_cache = {}

    def extract_grammar(self) -> Dict[z3.SortRef, List[Tuple[str, List[z3.SortRef]]]]:
        """
        Extracts the grammar from the synthesis problem.

        :return: A dictionary mapping each sort to a list of its constructors and their argument sorts.
        """
        grammar = {}
        for func_name, func_descriptor in self.context.z3_synth_functions.items():
            output_sort = func_descriptor.range()
            arg_sorts = [func_descriptor.domain(i) for i in range(func_descriptor.arity())]

            if output_sort not in grammar:
                grammar[output_sort] = []

            # need to decide if to allow recursive functions or not
            # grammar[output_sort].append((func_name, arg_sorts))

            # if output_sort == z3.IntSort():
            #     grammar[output_sort].extend([
            #         ("Plus", [z3.IntSort(), z3.IntSort()]),
            #         ("Minus", [z3.IntSort(), z3.IntSort()]),
            #         ("Neg", [z3.IntSort()]),
            #         ("Ite", [z3.BoolSort(), z3.IntSort(), z3.IntSort()])
            #     ])
            #     for const in range(self.MIN_CONST, self.MAX_CONST + 1):
            #         grammar[output_sort].append((f"Times{const}", [z3.IntSort()]))
            # 
            #     for const in range(self.MIN_CONST, self.MAX_CONST + 1):
            #         grammar[output_sort].append((f"Const_{const}", []))
            # 
            # elif output_sort == z3.BoolSort():
            #     grammar[output_sort].extend([
            #         ("LE", [z3.IntSort(), z3.IntSort()]),
            #         ("GE", [z3.IntSort(), z3.IntSort()]),
            #         ("LT", [z3.IntSort(), z3.IntSort()]),
            #         ("GT", [z3.IntSort(), z3.IntSort()]),
            #         ("Eq", [z3.IntSort(), z3.IntSort()]),
            #         ("And", [z3.BoolSort(), z3.BoolSort()]),
            #         ("Or", [z3.BoolSort(), z3.BoolSort()]),
            #         ("Not", [z3.BoolSort()]),
            #         ("Implies", [z3.BoolSort(), z3.BoolSort()]),
            #         ("Xor", [z3.BoolSort(), z3.BoolSort()])
            #    ])

        if z3.IntSort() not in grammar:
            grammar[z3.IntSort()] = []

        grammar[z3.IntSort()].extend([
            ("Plus", [z3.IntSort(), z3.IntSort()]),
            ("Minus", [z3.IntSort(), z3.IntSort()]),
            ("Neg", [z3.IntSort()]),
            ("Ite", [z3.BoolSort(), z3.IntSort(), z3.IntSort()])
        ])
        for const in range(self.MIN_CONST, self.MAX_CONST + 1):
            grammar[z3.IntSort()].append((f"Times{const}", [z3.IntSort()]))

        for const in range(self.MIN_CONST, self.MAX_CONST + 1):
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

    def compute_constructor_classes(self) -> Dict[z3.SortRef, List[Tuple[str, List[z3.SortRef]]]]:
        """
        Computes constructor classes for each sort in the grammar.

        :return: A dictionary mapping each sort to a list of its constructor classes.
        """
        constructor_classes = {}
        for sort, constructors in self.grammar.items():
            constructor_classes[sort] = []
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
            return []

        if (sort, size) in self.term_cache:
            return self.term_cache[(sort, size)]

        terms = []
        unique_terms = set()

        if size == 0:
            base_terms = self.get_base_terms(sort)
            terms.extend(base_terms)
            unique_terms.update(base_terms)

        queue = [(sort, size)]
        while queue:
            current_sort, current_size = queue.pop(0)
            if current_size == 0:
                continue

            for constructor_class in self.constructor_classes.get(current_sort, []):
                for constructor, arg_sorts in constructor_class:
                    for size_combination in self.generate_size_combinations(len(arg_sorts), current_size - 1):
                        arg_terms = []
                        for arg_sort, sub_size in zip(arg_sorts, size_combination):
                            if (arg_sort, sub_size) not in self.term_cache:
                                queue.append((arg_sort, sub_size))
                            arg_terms.append(self.term_cache.get((arg_sort, sub_size), []))

                        if constructor == 'Ite':
                            bool_terms = self.term_cache.get((z3.BoolSort(), size_combination[0]), [])
                            int_terms_1 = self.term_cache.get((z3.IntSort(), size_combination[1]), [])
                            int_terms_2 = self.term_cache.get((z3.IntSort(), size_combination[2]), [])
                            for bool_term, int_term_1, int_term_2 in product(bool_terms, int_terms_1, int_terms_2):
                                term = self.construct_term(constructor, (bool_term if bool_term not in (0,1) else True, int_term_1, int_term_2))
                                simplified_term = z3.simplify(term)
                                if simplified_term not in unique_terms:
                                    terms.append(simplified_term)
                                    unique_terms.add(simplified_term)
                        else:
                            for term_combination in product(*arg_terms):
                                term = self.construct_term(constructor, term_combination)
                                simplified_term = z3.simplify(term)
                                if simplified_term not in unique_terms:
                                    terms.append(simplified_term)
                                    unique_terms.add(simplified_term)

        self.term_cache[(sort, size)] = terms
        return terms

    def get_base_terms(self, sort: z3.SortRef) -> List[z3.ExprRef]:
        """
        Gets the base terms (variables) of the given sort.

        :param sort: The Z3 sort of the base terms.
        :return: A list of Z3 variables of the given sort.
        """
        return [var for var in self.context.z3_variables.values() if var.sort() == sort]

    def construct_term(self, constructor: str, term_combination: Tuple[z3.ExprRef | z3.ArithRef, ...]) -> z3.ExprRef:
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
        elif constructor == 'LE':
            return z3.If(term_combination[0] <= term_combination[1], z3.IntVal(1), z3.IntVal(0))
        elif constructor == 'GE':
            return z3.If(term_combination[0] >= term_combination[1], z3.IntVal(1), z3.IntVal(0))
        elif constructor == 'LT':
            return z3.If(term_combination[0] < term_combination[1], z3.IntVal(1), z3.IntVal(0))
        elif constructor == 'GT':
            return z3.If(term_combination[0] > term_combination[1], z3.IntVal(1), z3.IntVal(0))
        elif constructor == 'Eq':
            return z3.If(term_combination[0] == term_combination[1], z3.IntVal(1), z3.IntVal(0))
        elif constructor == 'And':
            return z3.And(term_combination[0], term_combination[1])
        elif constructor == 'Or':
            return z3.Or(term_combination[0], term_combination[1])
        elif constructor == 'Not':
            return z3.Not(term_combination[0])
        elif constructor == 'Implies':
            return z3.Implies(term_combination[0], term_combination[1])
        elif constructor == 'Xor':
            return z3.Xor(term_combination[0], term_combination[1])
        elif constructor == 'Ite':
            return z3.If(term_combination[0], term_combination[1], term_combination[2])
        else:
            func = self.context.z3_synth_functions.get(constructor)
            if func is None:
                raise ValueError(f"Unsupported constructor: {constructor}")
            return func(*term_combination)

    def generate_size_combinations(self, num_args: int, total_size: int) -> List[Tuple[int, ...]]:
        """
        Generates all possible combinations of argument sizes that sum up to the total size.

        :param num_args: The number of arguments.
        :param total_size: The total size of the arguments.
        :return: A list of tuples representing the size combinations.
        """
        combinations = [(0,) * num_args]
        for _ in range(total_size):
            new_combinations = []
            for combination in combinations:
                for i in range(num_args):
                    new_combination = list(combination)
                    new_combination[i] += 1
                    new_combinations.append(tuple(new_combination))
            combinations = new_combinations
        return combinations

    def generate(self, max_depth: int) -> Dict[z3.SortRef, List[z3.ExprRef]]:
        """
        Generates candidate terms up to the given maximum depth.

        :param max_depth: The maximum depth of the terms.
        :return: A dictionary mapping each sort to a list of generated terms.
        """
        generated_terms = {}
        for depth in range(max_depth + 1):
            for sort in self.grammar.keys():
                terms = self.fast_enum(sort, depth)
                if sort not in generated_terms:
                    generated_terms[sort] = []
                generated_terms[sort].append(terms)
        return generated_terms
