import json
import time
import subprocess
import csv
from datetime import datetime
from pathlib import Path
import uuid
import yaml
import psutil
from src.utilities.config_manager import ConfigManager

sygus_dir = "problems/sygus_comp_2019_clia_track"
output_cvc5_csv = "sygus_solver_cvc5_results_new.csv"
output_pysynthlab_csv = "sygus_solver_pysynthlab_results_new.csv"
project_root = Path(__file__).parent.parent.parent

py_synth_lab_solver_configs = [
    str(project_root / "src" / "config" / "benchmark_random_enumerative_bottom_up.yaml"),
    str(project_root / "src" / "config" / "benchmark_fast_enumerative.yaml"),
    str(project_root / "src" / "config" / "benchmark_random_weighted_top_down.yaml"),
    str(project_root / "src" / "config" / "benchmark_partial.yaml"),
    str(project_root / "src" / "config" / "benchmark_random_top_down.yaml"),
]

cvc5_configs = [
    "--sygus-enum=smart",
    "--sygus-enum=random",
    "--sygus-enum=fast",
    "--sygus-enum=var-agnostic",
    "--sygus-enum=smart --sygus-si=none",
    "--sygus-enum=smart --sygus-si=use",
    "--sygus-enum=smart --sygus-si=all",
    "--sygus-enum=smart --sygus-grammar-cons=simple",
    "--sygus-enum=smart --sygus-grammar-cons=any-term",
    "--sygus-enum=smart --sygus-fair=direct",
    "--sygus-enum=smart --sygus-fair=dt-size-bound",
    "--sygus-enum=smart --sygus-fair=none",
    "--sygus-enum=smart --sygus-repair-const",
]


def run_command_with_metrics(command, sampling_interval=0.1, timeout=None):
    start_time = time.time()
    print(f"Running command: {command}")
    process = psutil.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    cpu_usage = []
    memory_usage = []

    try:
        while process.poll() is None:
            try:
                cpu_percent = process.cpu_percent()
                memory_info = process.memory_info()

                cpu_usage.append(cpu_percent)
                memory_usage.append(memory_info.rss / 1024 / 1024)

            except psutil.NoSuchProcess:
                break

            time.sleep(sampling_interval)

            if timeout and (time.time() - start_time) > timeout:
                process.terminate()
                print(f"Process terminated due to timeout: {command}")
                break

        stdout, stderr = process.communicate()
        end_time = time.time()
        elapsed_time = end_time - start_time

        avg_cpu = sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0
        max_memory = max(memory_usage) if memory_usage else 0

        print(f"Command finished: {command}")
        return process.returncode, elapsed_time, stdout, stderr, avg_cpu, max_memory

    except Exception as e:
        print(f"An error occurred while running the command: {e}")
        return None, None, b"", str(e).encode(), None, None


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

    fieldnames = ["run_id", "run_datetime", "solver", "config", "file", "return_code", "time", "avg_cpu_usage", "max_memory_usage", "stdout", "stderr"]

    sygus_files = list(Path.joinpath(project_root, sygus_dir).glob("*.sl"))
    for sygus_file in sygus_files:
        print(f"Processing SyGuS file: {sygus_file}")

        for config in cvc5_configs:
            results = []
            print(f"Running cvc5 with config: {config}")
            command = f"cvc5 --tlimit={timeout_ms} {config} {sygus_file}"
            retcode, elapsed_time, stdout, stderr, avg_cpu, max_memory = run_command_with_metrics(command, timeout=timeout_ms/1000)
            results.append({
                "run_id": run_id,
                "run_datetime": run_datetime,
                "solver": "cvc5",
                "config": config,
                "file": str(sygus_file),
                "return_code": retcode,
                "time": elapsed_time,
                "avg_cpu_usage": avg_cpu,
                "max_memory_usage": max_memory,
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
    fieldnames += ["candidates_generated", "candidates_pruned", "iterations", "solution_height", "solution_complexity", "best_partial_score", "grammar_size", "solution_found", "avg_cpu_usage", "max_cpu_usage", "avg_per_core_cpu_usage", "max_per_core_cpu_usage", "max_memory_usage", "solution_space_coverage", "solver_calls", "avg_solver_time", "unique_patterns", "pattern_reuse_ratio"]

    sygus_files = list(Path.joinpath(project_root, sygus_dir).glob("*.sl"))
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
                problem=sygus_file.stem,
                root=project_root
            )
            config['logging']['metrics_file'] = config['logging']['metrics_file'].format(
                datetime=datetime.now().strftime("%Y%m%d_%H%M%S"),
                problem=sygus_file.stem,
                root=project_root
            )
            benchmark_name = Path(config_file).stem
            temp_config_file = f"temp_{benchmark_name}_config.yaml"
            with open(temp_config_file, 'w') as f:
                yaml.dump(config, f)

            metrics_file = config['logging']['metrics_file']
            command = str(project_root) + f"/pysynthlab/bin/python -m src.runner --config {Path.joinpath(Path.cwd(), temp_config_file)} {sygus_file}"
            retcode, elapsed_time, stdout, stderr = run_command(command)

            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

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
                **metrics
            })

            Path(temp_config_file).unlink()
            Path(metrics_file).unlink()

        write_csv_results(output_pysynthlab_csv, results, fieldnames)
        print(f"Finished writing results to {output_pysynthlab_csv}")
    print(f"All pysynthlab finished")


if __name__ == "__main__":
    for i in range(3):
        #run_pysynthlab_experiments()
        run_cvc5_experiments()
