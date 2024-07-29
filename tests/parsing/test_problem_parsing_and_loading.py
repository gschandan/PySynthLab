import unittest
from pathlib import Path

from src.cegis.z3.synthesis_problem_z3 import SynthesisProblemZ3
from src.utilities.options import Options


class TestSyGusParsing(unittest.TestCase):
    def setUp(self):
        self.options = Options()
        self.options.logging.level = "DEBUG"
        self.sygus_dir = Path(__file__).parent.parent.parent / "problems" / "sygus_comp_2019_clia_track"
        self.options = Options()

    def test_parse_sygus_files(self):
        files = list(self.sygus_dir.glob("*.sl"))
        print(f"Found {len(files)} .sl files")
        for sygus_file in self.sygus_dir.glob("*.sl"):
            print(f"\nProcessing file: {sygus_file.name}")
            with self.subTest(file=sygus_file.name):
                with open(sygus_file, "r") as f:
                    problem_content = f.read()

                problem = SynthesisProblemZ3(problem_content, self.options)

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