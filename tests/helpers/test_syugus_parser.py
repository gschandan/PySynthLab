import unittest
from pysynthlab.helpers.sygus_parser import sygus_problem_parser


class TestProgramParser(unittest.TestCase):
    """ Tests of the sygus_problem_parser class. """

    def setUp(self):
        """ Create instance of BlockParser. """
        self.parser = sygus_problem_parser

    def testImportValidFile(self):
        """ Test file parsing """
        self.assertEqual(
            self.parser().__str__(),
            ("(set-logic LIA)\n"
             "\n"
             "(synth-fun f ((x Int) (y Int)) Int)\n"
             "\n"
             "(declare-var x Int)\n"
             "(declare-var y Int)\n"
             "(constraint (= (f x y) (f y x)))\n"
             "(constraint (and (<= x (f x y)) (<= y (f x y))))")
        )
