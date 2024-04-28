import itertools

from z3 import *
import pyparsing

from pysynthlab.synthesis_problem import SynthesisProblem
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, FileType


def manual_loops():
    print("METHOD 1: string maniuplation")
    base_problem = """
        (declare-fun x () Int)
        (declare-fun y () Int)
        (assert(or (not(= (f x y) (f y x))) (not (and (<= x (f x y)) (<= y (f x y))))))
        """

    guesses = [
        "(define-fun f ((x Int) (y Int)) Int x)",  # Guess 1: f(x, y) = x
        "(define-fun f ((x Int) (y Int)) Int y)",  # Guess 2: f(x, y) = y
        "(define-fun f ((x Int) (y Int)) Int (ite (<= x y) x y))",  # Guess 3: f(x, y) = min(x, y)
        "(define-fun f ((x Int) (y Int)) Int (ite (<= x y) y x))",  # Guess 4: f(x, y) = max(x, y)
    ]

    def try_guess(base_problem, guess):
        smt_lib_str = guess + base_problem + "(check-sat)(get-model)"
        solver = z3.Solver()
        solver.from_string(smt_lib_str)
        print("SMT:", solver.to_smt2())
        if solver.check() == z3.sat:
            print(f"Guess '{guess}' is not valid, found counterexample:")
            print(solver.model())
            return False
        else:
            print(f"Guess '{guess}' is potentially correct, no counterexample found.")
            return True

    for guess in guesses:
        if try_guess(base_problem, guess):
            break

    print("METOD 2: python function substitution")

    def add_negated_constraints(solver, f_guess):
        x, y = Ints('x y')
        f_x_y = f_guess(x, y)
        f_y_x = f_guess(y, x)
        solver.add(Or(Not(f_x_y == f_y_x), Not(And(x <= f_x_y, y <= f_x_y))))

    def add_original_constraints(solver, f_guess):
        x, y = Ints('x y')
        f_x_y = f_guess(x, y)
        f_y_x = f_guess(y, x)
        solver.add(And((f_x_y == f_y_x), And(x <= f_x_y, y <= f_x_y)))

    guesses = [
        (lambda a, b: 0, "f(x, y) = 0"),
        (lambda a, b: a, "f(x, y) = x"),
        (lambda a, b: b, "f(x, y) = y"),
        (lambda a, b: a - b, "f(x, y) = x - y"),
        (lambda a, b: If(a <= b, b, a), "f(x, y) = max(x, y)"),
        (lambda a, b: If(a <= b, a, b), "f(x, y) = min(x, y)"),

    ]

    for guess, name in guesses:
        enumerator = Solver()
        add_negated_constraints(enumerator, guess)
        print("ENUMAERATOR:", enumerator.to_smt2())

        if enumerator.check() == sat:
            model = enumerator.model()
            print(
                f"Counterexample for guess {name}: x = {model.evaluate(Int('x'))}, y = {model.evaluate(Int('y'))}")

            verifier = Solver()
            add_original_constraints(verifier, guess)
            verifier.add(Int('x') == model[Int('x')], Int('y') == model[Int('y')])
            print("VERIFIER:", verifier.to_smt2())

            if verifier.check() == sat:
                print(f"Verification passed unexpectedly for guess {name}. Possible error in logic.")
            else:
                print(f"Verification failed for guess {name}, counterexample confirmed.")
        else:
            verifier = Solver()
            add_original_constraints(verifier, guess)
            print("VERIFIER:", verifier.to_smt2())
            if verifier.check() == sat:
                print(f"No counterexample found for guess {name}. Guess should be correct.")
            else:
                print(f"Verification failed unexpectedly for guess {name}. Possible error in logic.")
        print("-" * 50)

    print("-" * 100)
    print("METHOD 3: function-based constraint addition")

    def add_negated_constraints(solver, f_guess):
        x, y = Ints('x y')
        f_x_y = f_guess(x, y)
        f_y_x = f_guess(y, x)
        solver.add(Or(Not(f_x_y == f_y_x), Not(And(x <= f_x_y, y <= f_x_y))))

    def add_original_constraints(solver, f_guess):
        x, y = Ints('x y')
        f_x_y = f_guess(x, y)
        f_y_x = f_guess(y, x)
        solver.add(And((f_x_y == f_y_x), And(x <= f_x_y, y <= f_x_y)))

    def create_solver_with_config():
        solver = z3.Solver()
        solver.set('smt.macro_finder', True)
        return solver

    guesses = [
        (lambda a, b: 0, "f(x, y) = 0"),
        (lambda a, b: a, "f(x, y) = x"),
        (lambda a, b: b, "f(x, y) = y"),
        (lambda a, b: a - b, "f(x, y) = x - y"),
        (lambda a, b: If(a <= b, b, a), "f(x, y) = max(x, y)"),
        (lambda a, b: If(a <= b, a, b), "f(x, y) = min(x, y)"),
    ]

    for guess, name in guesses:
        enumerator = create_solver_with_config()
        add_negated_constraints(enumerator, guess)
        print("ENUMERATOR:", enumerator.to_smt2())

        if enumerator.check() == sat:
            model = enumerator.model()
            print(f"Counterexample for guess {name}: x = {model.evaluate(Int('x'))}, y = {model.evaluate(Int('y'))}")

            verifier = create_solver_with_config()
            add_original_constraints(verifier, guess)
            verifier.add(Int('x') == model[Int('x')], Int('y') == model[Int('y')])
            print("VERIFIER:", verifier.to_smt2())

            if verifier.check() == sat:
                print(f"Verification passed unexpectedly for guess {name}. Possible error in logic.")
            else:
                print(f"Verification failed for guess {name}, counterexample confirmed.")
        else:
            verifier = create_solver_with_config()
            add_original_constraints(verifier, guess)
            print("VERIFIER:", verifier.to_smt2())
            if verifier.check() == sat:
                print(f"No counterexample found for guess {name}. Guess should be correct.")
            else:
                print(f"Verification failed unexpectedly for guess {name}. Possible error in logic.")
        print("-" * 50)

    print("-" * 100)
    print("METHOD 4: More generic")

    def create_solver_with_config():
        solver = z3.Solver()
        solver.set('smt.macro_finder', False)
        return solver

    def add_constraints(solver, variables, f_guess, negated=False):
        var_objs = [Int(var) for var in variables]
        constraints = []
        for i, var1 in enumerate(var_objs):
            for var2 in var_objs[i + 1:]:
                f_var1_var2 = f_guess(var1, var2)
                f_var2_var1 = f_guess(var2, var1)
                if negated:
                    constraints.append(
                        Or(Not(f_var1_var2 == f_var2_var1), Not(And(var1 <= f_var1_var2, var2 <= f_var1_var2))))
                else:
                    constraints.append(And(f_var1_var2 == f_var2_var1, var1 <= f_var1_var2, var2 <= f_var1_var2))
        solver.add(*constraints)

    def print_model(model, variables):
        results = ", ".join(f"{var} = {model.evaluate(Int(var))}" for var in variables)
        print("Counterexample: ", results)

    variables = ['x', 'y']
    guesses = [
        (lambda a, b: a + b, "f(x, y) = x + y"),
        (lambda a, b: a - b, "f(x, y) = x - y"),
        (lambda a, b: 0, "f(x, y) = 0"),
        (lambda a, b: a, "f(x, y) = x"),
        (lambda a, b: b, "f(x, y) = y"),
        (lambda a, b: If(a <= b, b, a), "f(x, y) = max(x, y)"),
        (lambda a, b: If(a <= b, a, b), "f(x, y) = min(x, y)"),
    ]

    for guess, name in guesses:
        enumerator = create_solver_with_config()
        add_constraints(enumerator, variables, guess, negated=True)
        print("ENUMERATOR:", enumerator.to_smt2())

        if enumerator.check() == sat:
            model = enumerator.model()
            print_model(model, variables)

            verifier = create_solver_with_config()
            add_constraints(verifier, variables, guess, negated=False)
            for var in variables:
                verifier.add(Int(var) == model[Int(var)])
            print("VERIFIER:", verifier.to_smt2())

            if verifier.check() == sat:
                print(f"Verification passed unexpectedly for guess {name}. Possible error in logic.")
            else:
                print(f"Verification failed for guess {name}, counterexample confirmed.")
        else:
            verifier = create_solver_with_config()
            add_constraints(verifier, variables, guess, negated=False)
            print("VERIFIER:", verifier.to_smt2())
            if verifier.check() == sat:
                print(f"No counterexample found for guess {name}. Guess should be correct.")
            else:
                print(f"Verification failed unexpectedly for guess {name}. Possible error in logic.")
        print("-" * 50)

    print("METHOD 5: Even more generic")

    def add_constraints(solver, f_guess, condition, constraint):
        x, y = Ints('x y')
        f_x_y = f_guess(x, y)
        f_y_x = f_guess(y, x)
        solver.add(condition(f_x_y, f_y_x, x, y, constraint))

    def negated_condition(f_x_y, f_y_x, x, y, constraint):
        return Or(Not(f_x_y == f_y_x), Not(constraint(x, y, f_x_y)))

    def original_condition(f_x_y, f_y_x, x, y, constraint):
        return And(f_x_y == f_y_x, constraint(x, y, f_x_y))

    def original_constraint(x, y, f_x_y):
        return And(x <= f_x_y, y <= f_x_y)

    guesses = [
        (lambda a, b: 0, "f(x, y) = 0"),
        (lambda a, b: a, "f(x, y) = x"),
        (lambda a, b: b, "f(x, y) = y"),
        (lambda a, b: a - b, "f(x, y) = x - y"),
        (lambda a, b: If(a <= b, b, a), "f(x, y) = max(x, y)"),
        (lambda a, b: If(a <= b, a, b), "f(x, y) = min(x, y)"),
    ]

    for guess, name in guesses:
        enumerator = Solver()
        add_constraints(enumerator, guess, negated_condition, original_constraint)
        print("ENUMERATOR:", enumerator.to_smt2())

        if enumerator.check() == sat:
            model = enumerator.model()
            print(f"Counterexample for guess {name}: x = {model.evaluate(Int('x'))}, y = {model.evaluate(Int('y'))}")

            verifier = Solver()
            add_constraints(verifier, guess, original_condition, original_constraint)
            verifier.add(Int('x') == model[Int('x')], Int('y') == model[Int('y')])
            print("VERIFIER:", verifier.to_smt2())

            if verifier.check() == sat:
                print(f"Verification passed unexpectedly for guess {name}. Possible error in logic.")
            else:
                print(f"Verification failed for guess {name}, counterexample confirmed.")
        else:
            verifier = Solver()
            add_constraints(verifier, guess, original_condition, original_constraint)
            print("VERIFIER:", verifier.to_smt2())
            if verifier.check() == sat:
                print(f"No counterexample found for guess {name}. Guess should be correct.")
            else:
                print(f"Verification failed unexpectedly for guess {name}. Possible error in logic.")
        print("-" * 50)

    print("Method 6")

    def add_constraints(solver, variables, funcs, condition_func):
        solver.add(condition_func(variables, funcs))

    def original_constraints(variables, funcs):
        x, y = variables
        f_x_y, f_y_x = funcs(x, y)
        return And(f_x_y == f_y_x, x <= f_x_y, y <= f_x_y)

    def negated_constraints(variables, funcs):
        x, y = variables
        f_x_y, f_y_x = funcs(x, y)
        return Or(Not(f_x_y == f_y_x), Not(And(x <= f_x_y, y <= f_x_y)))

    def function_wrapper(f_guess):
        def compute(x, y):
            return f_guess(x, y), f_guess(y, x)

        return compute

    guesses = [
        (lambda a, b: 0, "f(x, y) = 0"),
        (lambda a, b: a, "f(x, y) = x"),
        (lambda a, b: b, "f(x, y) = y"),
        (lambda a, b: a - b, "f(x, y) = x - y"),
        (lambda a, b: If(a <= b, b, a), "f(x, y) = max(x, y)"),
        (lambda a, b: If(a <= b, a, b), "f(x, y) = min(x, y)"),

    ]

    variables = Int('x'), Int('y')

    for guess, name in guesses:
        enumerator = Solver()
        add_constraints(enumerator, variables, function_wrapper(guess), negated_constraints)
        print("ENUMERATOR:", enumerator.to_smt2())

        if enumerator.check() == sat:
            model = enumerator.model()
            x, y = variables
            print(f"Counterexample for guess {name}: x = {model[x]}, y = {model[y]}")

            verifier = Solver()
            add_constraints(verifier, variables, function_wrapper(guess), original_constraints)
            verifier.add(x == model[x], y == model[y])
            print("VERIFIER:", verifier.to_smt2())

            if verifier.check() == sat:
                print(f"Verification passed unexpectedly for guess {name}. Possible error in logic.")
            else:
                print(f"Verification failed for guess {name}, counterexample confirmed.")
        else:
            verifier = Solver()
            add_constraints(verifier, variables, function_wrapper(guess), original_constraints)
            print("VERIFIER:", verifier.to_smt2())
            if verifier.check() == sat:
                print(f"No counterexample found for guess {name}. Guess should be correct.")
            else:
                print(f"Verification failed unexpectedly for guess {name}. Possible error in logic.")
        print("-" * 50)

    print("Method 7")

    def create_variables(num_vars, var_type=Int):
        return [var_type(f'var{i}') for i in range(num_vars)]

    def add_constraints(solver, variables, funcs, condition_func):
        solver.add(condition_func(variables, funcs))

    def original_constraints(variables, funcs):
        results = funcs(*variables)
        conditions = []
        for f_x, f_y in results:
            conditions.append(And(f_x == f_y, *[v <= f_x for v in variables]))
        return And(*conditions)

    def negated_constraints(variables, funcs):
        results = funcs(*variables)
        conditions = []
        for f_x, f_y in results:
            conditions.append(Or(Not(f_x == f_y), Not(And(*[v <= f_x for v in variables]))))
        return Or(*conditions)

    def function_wrapper(f_guess):
        def compute(*args):
            return [(f_guess(*args), f_guess(*reversed(args)))]

        return compute

    guesses = [
        (lambda a, b: 0, "f(x, y) = 0"),
        (lambda a, b: a, "f(x, y) = x"),
        (lambda a, b: b, "f(x, y) = y"),
        (lambda a, b: a - b, "f(x, y) = x - y"),
        (lambda a, b: If(a <= b, b, a), "f(x, y) = max(x, y)"),
        (lambda a, b: If(a <= b, a, b), "f(x, y) = min(x, y)"),

    ]

    num_vars = 2
    variables = create_variables(num_vars, Int)

    for guess, name in guesses:
        enumerator = Solver()
        add_constraints(enumerator, variables, function_wrapper(guess), negated_constraints)
        print("ENUMERATOR:", enumerator.to_smt2())

        if enumerator.check() == sat:
            model = enumerator.model()
            var_vals = [model[v] for v in variables]
            print(f"Counterexample for guess {name}: ",
                  ', '.join(f"{v} = {val}" for v, val in zip(variables, var_vals)))

            verifier = Solver()
            add_constraints(verifier, variables, function_wrapper(guess), original_constraints)
            for var, val in zip(variables, var_vals):
                verifier.add(var == val)
            print("VERIFIER:", verifier.to_smt2())

            if verifier.check() == sat:
                print(f"Verification passed unexpectedly for guess {name}. Possible error in logic.")
            else:
                print(f"Verification failed for guess {name}, counterexample confirmed.")
        else:
            verifier = Solver()
            add_constraints(verifier, variables, function_wrapper(guess), original_constraints)
            print("VERIFIER:", verifier.to_smt2())
            if verifier.check() == sat:
                print(f"No counterexample found for guess {name}. Guess should be correct.")
            else:
                print(f"Verification failed unexpectedly for guess {name}. Possible error in logic.")
        print("-" * 50)

    print("method 8")

    f = Function('f', IntSort(), IntSort(), IntSort())
    args = Ints('x y')

    def setup_constraints():
        f_x_y = f(*args)
        f_y_x = f(*args)
        return [Or(Not(f_x_y == f_y_x), Not(And(x <= f_x_y, y <= f_x_y)))]

    solver = Solver()
    constraints = setup_constraints()
    for constraint in constraints:
        solver.add(constraint)

    def substitute_function(solver, func_def):
        new_constraints = [
            substitute(c, (f(*args), func_def(*args)), (f(*args[::-1]), func_def(*args[::-1]))) for c in constraints
        ]
        solver.reset()
        for c in new_constraints:
            solver.add(c)

    def guess_a(x, y):
        return x + y

    def guess_b(x, y):
        return x - y

    def guess_c(x, y):
        return If(x <= y, y, x)

    substitute_function(solver, guesses[0][0])
    if solver.check() == sat:
        print("SAT with guess A:", solver.model())
    else:
        print("UNSAT with guess A")

    substitute_function(solver, guess_b)
    if solver.check() == sat:
        print("SAT with guess B:", solver.model())
    else:
        print("UNSAT with guess B")

    substitute_function(solver, guess_c)
    if solver.check() == sat:
        print("SAT with guess B:", solver.model())
    else:
        print("UNSAT with guess B")

def main(args):
    manual_loops()
    # file = args.input_file.read()
    #
    # problem = SynthesisProblem(file, int(args.sygus_standard))
    # parsed_sygus_problem = problem.convert_sygus_to_smt()
    # problem.info()
    # print(parsed_sygus_problem)
    #
    # print("INITIAL enumerator_solver SMT: ", problem.enumerator_solver.to_smt2())
    # print("INITIAL verification_solver SMT: ", problem.verification_solver.to_smt2())
    #
    # print("-" * 100)
    # generator = problem.generate_candidate_functions(0, 3, 0)
    # for func in generator:
    #     print(func(["x", "y", "z"]))  # Assuming 'args' would be something like this
    #
    # problem.execute_cegis()

    # while not found_valid_candidate:
    #     try:
    #         candidate_expression = next(candidate_expressions)
    #     except StopIteration:
    #         depth += 1
    #         if depth > depth_limit:
    #             print("Depth limit reached without finding a valid candidate.")
    #             break
    #         candidate_expressions = problem.generate_linear_integer_expressions(depth)
    #         candidate_expression = next(candidate_expressions)
    #
    #     expression = problem.z3_func(*problem.func_args) == candidate_expression # (ite (<= x y) y x)
    #     if itr == 100:
    #         p = list(problem.z3variables.values())
    #         expression = problem.z3_func(*problem.func_args) == z3.If(p[0] <= p[1], p[1], p[0])
    #     if expression in problem.assertions:
    #         itr += 1
    #         continue
    #     print("expr:", expression)
    #
    #     problem.enumerator_solver.push()
    #     problem.enumerator_solver.add(expression)
    #     enumerator_solver_result = problem.enumerator_solver.check()
    #     print("Verification result:", enumerator_solver_result)
    #     problem.enumerator_solver.pop()
    #     model = problem.enumerator_solver.model()
    #     counterexample = problem.check_counterexample(model)
    #     if counterexample is not None:
    #         additional_constraint = problem.get_additional_constraints(counterexample)
    #         problem.enumerator_solver.add(additional_constraint)
    #     itr += 1
    #     print(f"Depth {depth}, Iteration {itr}")

    # if found_valid_candidate:
    #     print("VALID CANDIDATE:", expression)
    # else:
    #     print("No valid candidate found within the depth/loop/time limit.")

    # print("VERIFICATION SMT: ", problem.verification_solver.to_smt2())
    # print("COUNTEREXAMPLE SMT: ", problem.enumerator_solver.to_smt2())


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '-q', '--quiet', action='store_true',
        help='Suppress all messages and debugging output')

    parser.add_argument(
        '-s', '--sygus-standard', default='2', choices=['1', '2'],
        help='The SyGuS language standard used in the input file')

    parser.add_argument(
        'input_file', type=FileType('r'),
        help='Path to an input file (or stdin if "-")')

    main(parser.parse_args())
