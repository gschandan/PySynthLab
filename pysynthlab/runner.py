from cvc5 import *

from pysynthlab.synthesis_problem import SynthesisProblem
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, FileType

def main(args):
    # file = args.input_file.read()
    #
    # problem = SynthesisProblem(file, int(args.sygus_standard))
    # parsed_sygus_problem = problem.convert_sygus_to_smt()
    # problem.info()
    # print(parsed_sygus_problem)

    # Create solvers
    verifier = Solver()
    verifier.setLogic("UFLIA")
    verifier.setOption("produce-models", "true")
    verifier.setOption("produce-unsat-cores", "true")
    enumerator = Solver()
    enumerator.setLogic("UFLIA")
    enumerator.setOption("produce-models", "true")
    enumerator.setOption("produce-unsat-cores", "true")

    intSort = verifier.getIntegerSort()
    x = verifier.mkConst(intSort, "x")
    y = verifier.mkConst(intSort, "y")
    # Declare the uninterpreted function f of sort (Int, Int) -> Int
    f_sort = verifier.mkFunctionSort([intSort, intSort], intSort)  # Function sort from Int, Int to Int
    f = verifier.mkConst(f_sort, "f")

    # Create the constraints
    constraints = [
        verifier.mkTerm(Kind.EQUAL, verifier.mkTerm(Kind.APPLY_UF, f, x, y), verifier.mkTerm(Kind.APPLY_UF, f, y, x)),
        verifier.mkTerm(Kind.LEQ, x, verifier.mkTerm(Kind.APPLY_UF, f, x, y)),
        verifier.mkTerm(Kind.LEQ, y, verifier.mkTerm(Kind.APPLY_UF, f, x, y))
    ]
    # TODO: fix these
    negated_constraints = verifier.mkTerm(Kind.OR,
                                        verifier.mkTerm(Kind.NOT,
                                                      verifier.mkTerm(Kind.EQUAL, verifier.mkTerm(Kind.APPLY_UF, f, x, y),
                                                                    verifier.mkTerm(Kind.APPLY_UF, f, y, x))),
                                        verifier.mkTerm(Kind.OR,
                                                      verifier.mkTerm(Kind.GT, x, verifier.mkTerm(Kind.APPLY_UF, f, x, y)),
                                                      verifier.mkTerm(Kind.GT, y, verifier.mkTerm(Kind.APPLY_UF, f, y, x))
                                                      )
                                        )

    verifier.assertFormula(negated_constraints)

    print("Guess 1: f(x, y) = x")
    enumerator.push()
    f_x_y = enumerator.mkTerm(Kind.APPLY_UF, f, x, y)  # Correctly apply function 'f' to x and y
    enumerator.assertFormula(enumerator.mkTerm(Kind.EQUAL, f_x_y, x))
    if enumerator.checkSat().isSat():
        print("Candidate satisfies counterexamples")
        verifier.push()
        verifier.assertFormula(verifier.mkTerm(Kind.EQUAL, verifier.mkTerm(Kind.APPLY_UF, f, x, y), x))
        if verifier.checkSat().isSat():
            m = verifier.getModel([],[x,y,f])
            print("Counterexample found:")
            print(m)
            enumerator.pop()
            xVal = m.evaluate(x)
            yVal = m.evaluate(y)
            enumerator.assertFormula(
                enumerator.mkTerm(Kind.DISTINCT, enumerator.mkTerm(Kind.APPLY_UF, f, [xVal, yVal]),
                                  m.evaluate(enumerator.mkTerm(Kind.APPLY_UF, f, [xVal, yVal]))))
        verifier.pop()
    else:
        print("Candidate does not satisfy counterexamples")

    # # Guess 2: f as constant 0
    # print("Guess 2: f(x, y) = 0")
    # enumerator.push()
    # enumerator.assertFormula(
    #     enumerator.mkTerm(Kind.EQUAL, enumerator.mkTerm(Kind.APPLY_UF, f, x, y), enumerator.mkInteger(0)))
    # if enumerator.checkSat().isSat():
    #     print("Candidate satisfies counterexamples")
    #     verifier.push()
    #     verifier.assertFormula(
    #         verifier.mkTerm(Kind.EQUAL, verifier.mkTerm(Kind.APPLY_UF, f, x, y), verifier.mkInteger(0)))
    #     if verifier.checkSat().isSat():
    #         m = verifier.getModel([], constants)
    #         print("Counterexample found:")
    #         print(m)
    #         enumerator.pop()
    #         xVal = m.evaluate(x)
    #         yVal = m.evaluate(y)
    #         enumerator.assertFormula(
    #             enumerator.mkTerm(Kind.DISTINCT, enumerator.mkTerm(Kind.APPLY_UF, f, xVal, yVal),
    #                               m.evaluate(enumerator.mkTerm(Kind.APPLY_UF, f, xVal, yVal))))
    #     verifier.pop()
    # else:
    #     print("Candidate does not satisfy counterexamples")
    #
    # # Guess 3: correct solution
    # print("Guess 3: f(x, y) = If(x <= y, y, x)")
    # enumerator.push()
    # enumerator.assertFormula(enumerator.mkTerm(Kind.EQUAL, enumerator.mkTerm(Kind.APPLY_UF, f, x, y),
    #                                            enumerator.mkTerm(Kind.ITE, enumerator.mkTerm(Kind.LEQ, x, y), y,
    #                                                              x)))
    # if enumerator.checkSat().isSat():
    #     print("Candidate satisfies counterexamples")
    #     verifier.push()
    #     verifier.assertFormula(verifier.mkTerm(Kind.EQUAL, verifier.mkTerm(Kind.APPLY_UF, f, x, y),
    #                                            verifier.mkTerm(Kind.ITE, verifier.mkTerm(Kind.LEQ, x, y), y, x)))
    #     if verifier.checkSat().isSat():
    #         m = verifier.getModel([], constants)
    #         print("Counterexample found:")
    #         print(m)
    #         enumerator.pop()
    #         xVal = m.evaluate(x)
    #         yVal = m.evaluate(y)
    #         enumerator.assertFormula(
    #             enumerator.mkTerm(Kind.DISTINCT, enumerator.mkTerm(Kind.APPLY_UF, f, xVal, yVal),
    #                               m.evaluate(enumerator.mkTerm(Kind.APPLY_UF, f, xVal, yVal))))
    #     else:
    #         print("Valid solution found with the third guess")
    #     verifier.pop()
    # else:
    #     print("Candidate does not satisfy counterexamples")

    # depth = 0
    # itr = 0
    # depth_limit = 200
    # found_valid_candidate = False
    #
    # problem.verification_solver.add(z3.parse_smt2_string(parsed_sygus_problem))
    #
    # assertions = problem.verification_solver.assertions()
    # problem.assertions.update(assertions)
    # for assertion in assertions:
    #     problem.original_assertions.append(assertion)
    #
    # problem.enumerator_solver.reset()
    #
    # negated_assertions = problem.negate_assertions(assertions)
    # problem.enumerator_solver.add(*negated_assertions)
    # problem.negated_assertions.update(negated_assertions)
    #
    # #problem.counterexample_solver.set("timeout", 30000)
    # #problem.verification_solver.set("timeout", 30000)
    #
    # set_param("smt.random_seed", 1234)
    # candidate_expressions = problem.generate_linear_integer_expressions(depth)
    # expression = None
    #
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
    #
    # if found_valid_candidate:
    #     print("VALID CANDIDATE:", expression)
    # else:
    #     print("No valid candidate found within the depth/loop/time limit.")
    #
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
