import time
import subprocess
import csv
from datetime import datetime
from pathlib import Path
import uuid
import yaml
from src.utilities.config_manager import ConfigManager

# sygus_dir = "problems/debugging"
sygus_dir = "problems/sygus_comp_2019_clia_track"
output_cvc5_csv = "sygus_solver_cvc5_results.csv"
output_pysynthlab_csv = "sygus_solver_pysynthlab_results_2.csv"
project_root = Path(__file__).parent.parent.parent

py_synth_lab_solver_configs = [
    # str(project_root / "src" / "config" / "benchmark_random_enumerative_bottom_up.yaml"),
    # str(project_root / "src" / "config" / "benchmark_fast_enumerative.yaml"),
    str(project_root / "src" / "config" / "benchmark_partial.yaml"),
    # str(project_root / "src" / "config" / "benchmark_random_weighted_top_down.yaml"),
    # str(project_root / "src" / "config" / "benchmark_random_top_down.yaml"),
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


def write_csv_results(filename, results, fieldnames):
    file_exists = Path(filename).exists()
    mode = 'a' if file_exists else 'w'
    with open(filename, mode, newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for result in results:
            writer.writerow(result)


def run_cvc5_experiments(timeout_ms=30000):
    run_id = str(uuid.uuid4())
    run_datetime = datetime.now().isoformat()

    cvc5_configs = [
        f"--tlimit={timeout_ms} --sygus-enum=smart",
        f"--tlimit={timeout_ms} --sygus-enum=random",
        f"--tlimit={timeout_ms} --sygus-enum=fast",
        f"--tlimit={timeout_ms} --sygus-enum=var-agnostic",
        f"--tlimit={timeout_ms} --sygus-enum=smart --sygus-si=none",
        f"--tlimit={timeout_ms} --sygus-enum=smart --sygus-si=use",
        f"--tlimit={timeout_ms} --sygus-enum=smart --sygus-si=all",
        f"--tlimit={timeout_ms} --sygus-enum=smart --sygus-abort-size=10",
        f"--tlimit={timeout_ms} --sygus-enum=smart --sygus-abort-size=20",
        f"--tlimit={timeout_ms} --sygus-enum=smart --sygus-grammar-cons=simple",
        f"--tlimit={timeout_ms} --sygus-enum=smart --sygus-grammar-cons=any-term",
        f"--tlimit={timeout_ms} --sygus-enum=smart --sygus-fair=direct",
        f"--tlimit={timeout_ms} --sygus-enum=smart --sygus-fair=dt-size-bound",
        f"--tlimit={timeout_ms} --sygus-enum=smart --sygus-fair=none",
        f"--tlimit={timeout_ms} --sygus-enum=smart --sygus-repair-const",
    ]

    fieldnames = ["run_id", "run_datetime", "solver", "config", "file", "return_code", "time", "stdout", "stderr"]

    sygus_files = list(Path(sygus_dir).glob("*.sl"))
    for sygus_file in sygus_files:
        results = []
        print(f"Processing SyGuS file: {sygus_file}")

        for config in cvc5_configs:
            print(f"Running cvc5 with config: {config}")
            command = f"cvc5 {config} {sygus_file}"
            retcode, elapsed_time, stdout, stderr = run_command(command)
            results.append({
                "run_id": run_id,
                "run_datetime": run_datetime,
                "solver": "cvc5",
                "config": config,
                "file": str(sygus_file),
                "return_code": retcode,
                "time": elapsed_time,
                "stdout": stdout.decode("utf-8"),
                "stderr": stderr.decode("utf-8"),
            })

        write_csv_results(output_cvc5_csv, results, fieldnames)
        print(f"Finished writing results to {output_cvc5_csv}")
    print(f"All cvc5 finished")


def run_pysynthlab_experiments():
    run_id = str(uuid.uuid4())
    run_datetime = datetime.now().isoformat()

    fieldnames = ["run_id", "run_datetime", "solver", "config", "file", "return_code", "time", "stdout", "stderr"]

    sygus_files = list(Path(sygus_dir).glob("*.sl"))
    for sygus_file in sygus_files:
        results = []
        print(f"Processing SyGuS file: {sygus_file}")

        for config_file in py_synth_lab_solver_configs:
            print(f"Running PySynthLab with config file: {config_file}")
            config = ConfigManager.load_yaml(config_file)
            if config is None:
                print(f"Skipping {config_file} due to missing configuration")
                continue
            config['logging']['file'] = config['logging']['file'].format(
                datetime=datetime.now().strftime("%Y%m%d_%H%M%S"),
                problem=sygus_file.stem
            )

            benchmark_name = Path(config_file).stem
            temp_config_file = f"temp_{benchmark_name}_config.yaml"
            with open(temp_config_file, 'w') as f:
                yaml.dump(config, f)

            command = f"./venv/bin/python -m src.runner --config {temp_config_file} {sygus_file}"
            retcode, elapsed_time, stdout, stderr = run_command(command)
            results.append({
                "run_id": run_id,
                "run_datetime": run_datetime,
                "solver": "PySynthLab",
                "config": config_file,
                "file": str(sygus_file),
                "return_code": retcode,
                "time": elapsed_time,
                "stdout": stdout.decode("utf-8"),
                "stderr": stderr.decode("utf-8"),
            })

            Path(temp_config_file).unlink()

        write_csv_results(output_pysynthlab_csv, results, fieldnames)
        print(f"Finished writing results to {output_pysynthlab_csv}")
    print(f"All pysynthlab finished")


if __name__ == "__main__":
    for i in range(1):
        #run_cvc5_experiments()
        run_pysynthlab_experiments()
