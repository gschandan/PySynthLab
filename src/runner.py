from dataclasses import asdict

from z3 import Int, Function, IntSort, Solver, ForAll, Implies, sat

from src.cegis.z3.cegis_t import CegisT
from src.cegis.z3.config_manager import ConfigManager
from src.cegis.z3.synthesis_problem import SynthesisProblem


def main() -> None:
    config = ConfigManager.get_config()
    print(asdict(config))

    problem = """
    (set-logic LIA)
    (synth-fun max2 ((a Int) (b Int)) Int)

    (declare-var x Int)
    (declare-var y Int)

    (constraint (>= (max2 x y) x))
    (constraint (>= (max2 x y) y))
    (constraint (or (= x (max2 x y)) (= y (max2 x y))))
    (constraint (= (max2 x x) x))

    (constraint (forall ((x Int) (y Int))
      (=> (>= x y) (= (max2 x y) x))))
    (constraint (forall ((x Int) (y Int))
      (=> (>= y x) (= (max2 y x) y))))
    (check-synth)
    """

    problem = SynthesisProblem(problem, config)
    print(problem.info_smt())
    # 
    # def synthesize(counterexamples):
    #     a = Int('a')
    #     b = Int('b')
    #     f = Function('f', IntSort(), IntSort(), IntSort())
    # 
    #     s = Solver()
    # 
    #     s.add(ForAll([a, b], Implies(a >= b, f(a, b) == a)))
    #     s.add(ForAll([a, b], Implies(b > a, f(a, b) == b)))
    # 
    #     for (x, y) in counterexamples:
    #         s.add(f(x, y) == max(x, y))
    # 
    #     if s.check() == sat:
    #         m = s.model()
    #         return m
    #     else:
    #         return None
    # 
    # def verify(candidate_solution):
    #     f = candidate_solution[0]
    #     for a in range(-10, 11):
    #         for b in range(-10, 11):
    #             expected = max(a, b)
    #             if candidate_solution[1](a, b) != expected:
    #                 return (a, b)
    #     return None
    # 
    # def extract_function(model):
    #     f = model[model.decls()[0]]
    #     return lambda a, b: f(a, b).as_long()
    # 
    # def cegis():
    #     counterexamples = []
    #     solution_found = False
    # 
    #     while not solution_found:
    #         candidate_solution = synthesize(counterexamples)
    # 
    #         if not candidate_solution:
    #             raise Exception("Unable to find a candidate solution")
    # 
    #         f = extract_function(candidate_solution)
    # 
    #         counterexample = verify((candidate_solution, f))
    # 
    #         if counterexample is None:
    #             solution_found = True
    #             return f
    #         else:
    #             counterexamples.append(counterexample)
    #solution = cegis()

    strategy = CegisT(problem)
    strategy.execute_cegis()

    # if config.synthesis_parameters_strategy == 'fast_enumerative':
    #     strategy = FastEnumerativeSynthesis(problem)
    # elif config.synthesis_parameters_strategy == 'random_search_bottom_up':
    #     strategy = RandomSearchStrategyBottomUp(problem)
    # elif config.synthesis_parameters_strategy == 'random_search_top_down':
    #     strategy = RandomSearchStrategyTopDown(problem)
    # elif config.synthesis_parameters_strategy == 'cegis_t_bottom_up':
    #     strategy = RandomSearchStrategyBottomUpCegisT(problem)
    # else:
    #     strategy = RandomSearchStrategyBottomUp(problem)
    # else:
    #     raise ValueError(f"Unknown synthesis strategy: {config.synthesis_parameters_strategy}")
    # 
    # print(strategy.problem.info_smt())
    # strategy.execute_cegis()


if __name__ == '__main__':
    main()
