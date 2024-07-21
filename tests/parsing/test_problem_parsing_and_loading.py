import os
import unittest
from pathlib import Path

from src.cegis.z3.synthesis_problem import SynthesisProblem
from src.utilities.options import Options

#@unittest.skip("optimising")
class TestSyGusParsing(unittest.TestCase):
    def setUp(self):
        self.options = Options()
        self.options.logging.level = "DEBUG"
        project_root = Path(__file__).parent.parent.parent 
        self.sygus_dir = project_root / "problems" / "sygus_comp_2019_clia_track"
        print(f"Looking for files in: {self.sygus_dir.absolute()}")
        self.options = Options()

    def test_parse_sygus_files(self):
        print(f"Looking for files in: {self.sygus_dir.absolute()}")
        files = list(self.sygus_dir.glob("*.sl"))
        print(f"Found {len(files)} .sl files")
        for sygus_file in self.sygus_dir.glob("*.sl"):
            print(f"\nProcessing file: {sygus_file.name}")
            with self.subTest(file=sygus_file.name):
                with open(sygus_file, "r") as f:
                    problem_content = f.read()

                problem = SynthesisProblem(problem_content, self.options)

                self.assertIsNotNone(problem.context, f"Failed to create context for {sygus_file.name}")

                self.assertGreater(len(problem.context.z3_variables), 0,
                                   f"No Z3 variables found in {sygus_file.name}")

                self.assertGreater(len(problem.context.z3_synth_functions), 0,
                                   f"No Z3 synth functions found in {sygus_file.name}")

                self.assertGreater(len(problem.context.z3_constraints), 0,
                                   f"No Z3 constraints found in {sygus_file.name}")

                self.assertNotEqual(problem.context.smt_problem, "",
                                    f"SMT problem is empty for {sygus_file.name}")

                self.assertGreater(len(problem.context.all_z3_functions), 0,
                                   f"No Z3 functions found in {sygus_file.name}")


if __name__ == "__main__":
    unittest.main()