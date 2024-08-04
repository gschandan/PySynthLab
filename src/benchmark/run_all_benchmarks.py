import time
import subprocess
import csv
from datetime import datetime
from pathlib import Path

import yaml

from src.utilities.config_manager import ConfigManager

sygus_dir = "problems/debugging"
output_cvc5_csv = "sygus_solver_cvc5_results.csv"
output_pysynthlab_csv = "sygus_solver_pysynthlab_results.csv"

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
    "--sygus-enum=smart --sygus-rr=none",
    "--sygus-enum=smart --sygus-rr=all",
    "--sygus-enum=smart --sygus-abort-size=10",
    "--sygus-enum=smart --sygus-abort-size=20",
    "--sygus-enum=smart --sygus-grammar-cons=min",
    "--sygus-enum=smart --sygus-grammar-cons=max",
    "--sygus-enum=smart --sygus-fair=direct",
    "--sygus-enum=smart --sygus-fair=dt-height",
]

py_synth_lab_solver_configs = [
    "config/benchmark_random_enumerative_bottom_up.yaml",
    "config/benchmark_random_top_down.yaml",
    "config/benchmark_random_weighted_topdown.yaml",
    "config/benchmark_fast_enumerative.yaml",
    "config/benchmark_partial.yaml",
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


def run_cvc5_experiments():
    with open(output_cvc5_csv, "w", newline="") as csvfile:
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

        with open(output_cvc5_csv, "a", newline="") as csvfile:
            fieldnames = ["solver", "config", "file", "return_code", "time", "stdout", "stderr"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            for result in results:
                writer.writerow(result)
        print(f"Finished writing results to {output_cvc5_csv}")
    print(f"All cvc5 finished")


def run_pysynthlab_experiments():
    with open(output_pysynthlab_csv, "w", newline="") as csvfile:
        fieldnames = ["solver", "config", "file", "return_code", "time", "stdout", "stderr"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    sygus_files = list(Path(sygus_dir).glob("*.sl"))
    for sygus_file in sygus_files:
        results = []
        print(f"Processing SyGuS file: {sygus_file}")

        for config_file in py_synth_lab_solver_configs:
            print(f"Running PySynthLab with config file: {config_file}")
            config = ConfigManager.load_yaml(config_file)

            benchmark_name = Path(config_file).stem
            config['logging']['file'] = config['logging']['file'].format(
                datetime=datetime.now().strftime("%Y%m%d_%H%M%S")
            )

            temp_config_file = f"temp_{benchmark_name}_config.yaml"
            with open(temp_config_file, 'w') as f:
                yaml.dump(config, f)

            command = f"./venv/bin/python -m src.runner --config {temp_config_file} {sygus_file}"
            retcode, elapsed_time, stdout, stderr = run_command(command)
            results.append({
                "solver": "PySynthLab",
                "config": config_file,
                "file": str(sygus_file),
                "return_code": retcode,
                "time": elapsed_time,
                "stdout": stdout.decode("utf-8"),
                "stderr": stderr.decode("utf-8"),
            })

            Path(temp_config_file).unlink()

        print(f"Writing results to {output_pysynthlab_csv}")
        with open(output_pysynthlab_csv, "a", newline="") as csvfile:
            fieldnames = ["solver", "config", "file", "return_code", "time", "stdout", "stderr"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            for result in results:
                writer.writerow(result)
        print(f"Finished writing results to {output_pysynthlab_csv}")
    print(f"All pysynthlab finished")


if __name__ == "__main__":
    run_cvc5_experiments()
    #run_pysynthlab_experiments()