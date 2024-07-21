import unittest
from tests.debugging.specific_problems.four_synth_functions import FourSynthFunctions
from tests.debugging.specific_problems.loop_problem import LoopProblem
from tests.debugging.specific_problems.max_three_integers import MaxThreeIntegers
from tests.debugging.specific_problems.max_two_integers import MaxTwoIntegers
from tests.debugging.specific_problems.predefined_func_substitutions import SubstitutionCheck

if __name__ == '__main__':
    test_loader = unittest.TestLoader()

    test_suite = unittest.TestSuite()

    test_suite.addTest(test_loader.loadTestsFromTestCase(MaxThreeIntegers))
    test_suite.addTest(test_loader.loadTestsFromTestCase(MaxTwoIntegers))
    test_suite.addTest(test_loader.loadTestsFromTestCase(FourSynthFunctions))
    test_suite.addTest(test_loader.loadTestsFromTestCase(LoopProblem))
    test_suite.addTest(test_loader.loadTestsFromTestCase(SubstitutionCheck))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    exit(not result.wasSuccessful())
