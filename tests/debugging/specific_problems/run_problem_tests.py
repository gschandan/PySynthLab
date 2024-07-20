import unittest
from tests.debugging.specific_problems.test_max_three_integers import MaxThreeIntegers
from tests.debugging.specific_problems.test_max_two_integers import MaxTwoIntegers
from tests.debugging.specific_problems.test_four_synth_functions import FourSynthFunctions
from tests.debugging.specific_problems.test_loop_problem import LoopProblem
from tests.debugging.specific_problems.test_predefined_func_substitutions import SubstitutionCheck

if __name__ == '__main__':
    test_loader = unittest.TestLoader()

    test_suite = unittest.TestSuite()

    test_suite.addTest(test_loader.loadTestsFromTestCase(MaxThreeIntegers))
    test_suite.addTest(test_loader.loadTestsFromTestCase(MaxTwoIntegers))
    test_suite.addTest(test_loader.loadTestsFromTestCase(FourSynthFunctions))
    test_suite.addTest(test_loader.loadTestsFromTestCase(LoopProblem))
    test_suite.addTest(test_loader.loadTestsFromTestCase(SubstitutionCheck))

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(test_suite)