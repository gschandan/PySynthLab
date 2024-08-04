import time
import subprocess
import csv
from pathlib import Path

sygus_dir = "problems/debugging"
output_csv = "sygus_solver_comparison.csv"

cvc5_configs = [
    "--sygus-enum=smart",
    "--sygus-enum=random",
    "--sygus-enum=fast",
    "--sygus-enum=var-agnostic",
    "--sygus-enum=smart --sygus-si=none",
    "--sygus-enum=random --sygus-si=none",
    "--sygus-enum=fast --sygus-si=none",
    "--sygus-enum=var-agnostic --sygus-si=none",
    "--sygus-enum=smart --sygus-si=use",
    "--sygus-enum=random --sygus-si=use",
    "--sygus-enum=fast --sygus-si=use",
    "--sygus-enum=var-agnostic --sygus-si=use",
    "--sygus-enum=smart --sygus-si=all",
    "--sygus-enum=random --sygus-si=all",
    "--sygus-enum=fast --sygus-si=all",
    "--sygus-enum=var-agnostic --sygus-si=all",
]


py_synth_lab_solver_configs = [
    "--synthesis_parameters__strategy=random_enumerative --synthesis_parameters__candidate_generation=bottom_up",
    "--synthesis_parameters__strategy=partial",
    # "--synthesis_parameters__strategy=random_enumerative --synthesis_parameters__candidate_generation=top_down",
    # "--synthesis_parameters__strategy=fast_enumerative",
]


def run_command(command):
    start_time = time.time()
    print(f"Running command: {command}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Command finished: {command}")
    return process.returncode, elapsed_time, stdout, stderr


def run_experiments():
    results = []
    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = ["solver", "config", "file", "return_code", "time", "stdout", "stderr"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

    sygus_files = list(Path(sygus_dir).glob("*.sl"))
    for sygus_file in sygus_files:
        results = []
        print(f"Processing SyGuS file: {sygus_file}")
        for config in cvc5_configs:
            print(f"Running cvc5 with config: {config}")
            command = f"cvc5 {config} {sygus_file}"
            retcode, elapsed_time, stdout, stderr = run_command(command)
            results.append({
                "solver": "cvc5",
                "config": config,
                "file": str(sygus_file),
                "return_code": retcode,
                "time": elapsed_time,
                "stdout": stdout.decode("utf-8"),
                "stderr": stderr.decode("utf-8"),
            })

        for config in py_synth_lab_solver_configs:
            print(f"Running my_solver with config: {config}")
            command = f"./venv/bin/python -m src.runner {config} {sygus_file}"
            retcode, elapsed_time, stdout, stderr = run_command(command)
            results.append({
                "solver": "my_solver",
                "config": config,
                "file": str(sygus_file),
                "return_code": retcode,
                "time": elapsed_time,
                "stdout": stdout.decode("utf-8"),
                "stderr": stderr.decode("utf-8"),
            })

        print(f"Writing results to {output_csv}")
        with open(output_csv, "w", newline="") as csvfile:
            fieldnames = ["solver", "config", "file", "return_code", "time", "stdout", "stderr"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            for result in results:
                writer.writerow(result)
    print(f"Finished writing results to {output_csv}")


if __name__ == "__main__":
    run_experiments()