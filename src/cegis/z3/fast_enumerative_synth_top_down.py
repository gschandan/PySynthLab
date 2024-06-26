from itertools import product
from typing import Dict, List, Tuple, Union, Set

import z3

from src.cegis.z3.synthesis_problem import SynthesisProblem
from src.cegis.z3.synthesis_strategy import SynthesisStrategy


# http://homepage.divms.uiowa.edu/~ajreynol/cav19b.pdf
class FastEnumerativeSynthesisTopDown(SynthesisStrategy):

    def __init__(self, problem: SynthesisProblem):
        self.problem = problem
        self.min_const = self.problem.options.min_const
        self.max_const = self.problem.options.max_const
        self.grammar = self.extract_grammar()
        self.term_cache = {}

    def extract_grammar(self) -> Dict[z3.SortRef, List[Tuple[str, List[z3.SortRef]]]]:
        """
        Extracts the grammar from the synthesis problem.

        :return: A dictionary mapping each sort to a list of its constructors and their argument sorts.
        """
        grammar = {}
        for func_name, func_descriptor in self.problem.context.z3_synth_functions.items():
            output_sort = func_descriptor.range()

            if output_sort not in grammar:
                grammar[output_sort] = []

            # need to decide if to allow recursive functions or not
            # arg_sorts = [func_descriptor.domain(i) for i in range(func_descriptor.arity())]
            # grammar[output_sort].append((func_name, arg_sorts))

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

    def top_down_enumerate(self, expected_sort: z3.SortRef):
        c_list = set()
        r_list = set(self.grammar.keys())
        depth = 0

        while depth <= self.problem.options.max_depth:
            print(f"Exploring depth: {depth}")
            new_c_list = set()

            for r in r_list:
                for t in self.get_terminals(r):
                    if self.can_expand(r, t):
                        new_terms = self.expand(c_list, t)
                        new_c_list.update(new_terms)

                for r_prime in self.grammar.keys():
                    if self.can_expand(r, r_prime):
                        r_list = self.expand(r_list, r_prime)

            c_list.update(new_c_list)
            c_list = self.eliminate_equivalents(c_list)

            for c in c_list:
                if c.sort() == expected_sort:
                    if self.is_correct(c):
                        return c
                    else:
                        print(f"Candidate failed: {c}")

            depth += 1

        return None

    def get_terminals(self, sort: z3.SortRef) -> List[Tuple[str, List[z3.SortRef]]]:
        return [rule for rule in self.grammar[sort] if not rule[1]]

    def can_expand(self, source: z3.SortRef, target: Union[z3.SortRef, Tuple[str, List[z3.SortRef]]]) -> bool:
        if isinstance(target, z3.SortRef):
            return source == target
        else:
            constructor, arg_sorts = target
            return source in self.grammar and all(arg_sort in self.grammar for arg_sort in arg_sorts)

    def expand(self, lst: Set, item: Union[z3.SortRef, Tuple[str, List[z3.SortRef]]]) -> Set:
        if isinstance(item, z3.SortRef):
            return lst.union({item})
        else:
            constructor, arg_sorts = item
            if not arg_sorts:
                term = self.construct_term(constructor, ())
                return lst.union({term})

            expanded_terms = []
            for arg_sort in arg_sorts:
                sub_terms = [term for term in lst if term.sort() == arg_sort]
                expanded_terms.append(sub_terms)

            new_terms = set()
            for args in product(*expanded_terms):
                try:
                    term = self.construct_term(constructor, args)
                    new_terms.add(term)
                except Exception:
                    pass

            return new_terms

    def eliminate_equivalents(self, terms: Set[z3.ExprRef]) -> Set[z3.ExprRef]:
        unique_terms = set()
        for term in terms:
            simplified_term = z3.simplify(term)
            if simplified_term not in unique_terms:
                unique_terms.add(simplified_term)
        return unique_terms

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
        elif constructor in ["LE", "GE", "LT", "GT", "Eq"]:
            arg1, arg2 = term_combination
            return z3.If(
                arg1 <= arg2 if constructor == "LE" else
                arg1 >= arg2 if constructor == "GE" else
                arg1 < arg2 if constructor == "LT" else
                arg1 > arg2 if constructor == "GT" else
                arg1 == arg2,
                True,
                False
            )
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
            cond, true_branch, false_branch = term_combination
            if not isinstance(cond, z3.BoolRef):
                raise ValueError(
                    f"Invalid condition type in 'Ite' constructor: {type(cond)}. "
                    f"Value: {cond}. Expected a Z3 BoolRef."
                )
            return z3.If(cond, true_branch, false_branch)
        else:
            func = self.problem.context.z3_synth_functions.get(constructor)
            if func is None:
                raise ValueError(f"Unsupported constructor: {constructor}")
            return func(*term_combination)

    def is_correct(self, candidate: z3.ExprRef) -> bool:
        for func_name, func in self.problem.context.z3_synth_functions.items():
            free_variables = list(self.problem.context.variable_mapping_dict[func_name].keys())
            func_str = f"{func_name}({', '.join([var.__str__() for var in free_variables])}) return {candidate.sexpr()}"
            self.problem.print_msg(f"Testing candidate {func_str}")
            result = self.test_candidates([func_str], [candidate])
            if result:
                return True
        return False
    
    def generate_candidates(self) -> List[Tuple[z3.ExprRef, str]]:
        pass
    
    def execute_cegis(self) -> None:
        for func_name, func in self.problem.context.z3_synth_functions.items():
            expected_sort = func.range()
            solution = self.top_down_enumerate(expected_sort)

            if solution:
                self.problem.print_msg(f"Found solution for function {func_name}: {solution.__str__()}", level=2)
                return

        self.problem.print_msg("No solution found", level=2)
