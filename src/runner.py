import sys
from dataclasses import asdict
from itertools import product

from z3 import *

from src.utilities.config_manager import ConfigManager
from src.cegis.z3.synthesis_strategy.fast_enumerative_synth import FastEnumerativeSynthesis
from src.cegis.z3.synthesis_strategy.random_search_bottom_up import RandomSearchStrategyBottomUp
from src.cegis.z3.synthesis_strategy.random_search_top_down import RandomSearchStrategyTopDown
from src.cegis.z3.synthesis_problem import SynthesisProblem


def main() -> None:
    def top_down_search(grammar, specification, max_depth=5):
        solver = Solver()

        def expand(expr, depth):
            print(f"{'  ' * depth}Expanding: {expr}")

            if depth > max_depth:
                print(f"{'  ' * depth}Max depth reached, backtracking...")
                return []

            expansions = []

            if expr in grammar:
                print(f"{'  ' * depth}Non-terminal {expr}, expanding productions:")
                for production in grammar[expr]:
                    print(f"{'  ' * depth}  Trying production: {production}")
                    expanded = expand(production, depth + 1)
                    expansions.extend(expanded)

            elif isinstance(expr, str):
                print(f"{'  ' * depth}Terminal {expr}, creating Z3 variable")
                return [Int(expr)]

            elif isinstance(expr, tuple):
                op, *args = expr
                print(f"{'  ' * depth}Expanding operation: {op}")
                arg_expansions = [expand(arg, depth + 1) for arg in args]

                for arg_combo in product(*arg_expansions):
                    if op == '+':
                        expansions.append(arg_combo[0] + arg_combo[1])
                    elif op == '-':
                        expansions.append(arg_combo[0] - arg_combo[1])
                    elif op == '*':
                        expansions.append(arg_combo[0] * arg_combo[1])
                    elif op == 'ite':
                        expansions.append(If(arg_combo[0], arg_combo[1], arg_combo[2]))
                    elif op == '>':
                        expansions.append(arg_combo[0] > arg_combo[1])
                    elif op == '>=':
                        expansions.append(arg_combo[0] >= arg_combo[1])
                    elif op == '<=':
                        expansions.append(arg_combo[0] <= arg_combo[1])
                    elif op == '<':
                        expansions.append(arg_combo[0] < arg_combo[1])

            print(f"{'  ' * depth}Expansions: {expansions}")
            return expansions

        print("Starting top-down search")
        candidates = expand('S', 0)

        for candidate in candidates:
            print(f"Checking candidate: {candidate}")
            solver.push()
            solver.add(specification(candidate))

            if solver.check() == sat:
                model = solver.model()
                print(f"Solution found: {candidate}")
                print(f"Model: {model}")
                return candidate, model

            solver.pop()

        print("No solution found")
        return None, None

    grammar = {
        'S': [('ite', 'B', 'T', 'T')],
        'B': [('>', 'T', 'T'), ('>=', 'T', 'T')],
        'T': ['x', 'y']
    }
    def specification(expr):
        x, y = Ints('x y')
        return ForAll([x, y],
                      Implies(And(x >= -10, x <= 10, y >= -10, y <= 10),
                              And(expr >= x, expr >= y, Or(expr == x, expr == y))
                              )
                      )
    
    result, model = top_down_search(grammar, specification)

    if result is not None:
        print(f"Solution found: {result}")

        x, y = Ints('x y')
        s = Solver()
        s.add(x >= -10, x <= 10, y >= -10, y <= 10)
        s.push()
        s.add(result != If(x > y, x, y))
        print("Solution is correct:", s.check() == unsat)
        s.pop()
        s.push()
        s.add(result != If(x >= y, x, y))
        print("Solution is also equivalent to If(x >= y, x, y):", s.check() == unsat)
        s.pop()
    else:
        print("No solution found")

    # config = ConfigManager.get_config()
    # ConfigManager.logger.info(asdict(config))
    # 
    # if config.input_source == 'stdin':
    #     problem_input = sys.stdin.read()
    # else:
    #     with open(config.input_source, 'r') as file:
    #         problem_input = file.read()
    # 
    # problem = SynthesisProblem(problem_input, config)
    # problem.logger.info(problem.info_smt())
    # 
    # if config.synthesis_parameters.strategy == 'fast_enumerative':
    #     strategy = FastEnumerativeSynthesis(problem)
    # elif config.synthesis_parameters.strategy == 'random_enumerative':
    #     if config.synthesis_parameters.candidate_generation == 'top_down':
    #         strategy = RandomSearchStrategyTopDown(problem)
    #     else:
    #         strategy = RandomSearchStrategyBottomUp(problem)
    # else:
    #     ConfigManager.logger.error(f"Unknown synthesis strategy: {config.synthesis_parameters.strategy}")
    #     raise ValueError(f"Unknown synthesis strategy: {config.synthesis_parameters.strategy}")
    # 
    # strategy.execute_cegis()


if __name__ == '__main__':
    main()
