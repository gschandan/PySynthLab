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


output_pysynthlab_csv = "sygus_solver_pysynthlab_results_partial.csv"
project_root = Path(__file__).parent.parent.parent

py_synth_lab_solver_configs = [
    str(project_root / "src" / "config" / "benchmark_partial.yaml"),
]
sygus_probs = [
    str(project_root / "problems" /"sygus_comp_2019_clia_track" /"jmbl_fg_polynomial1.sl"),
    str(project_root / "problems" /"sygus_comp_2019_clia_track" /"jmbl_fg_max2.sl"),
    str(project_root / "problems" /"sygus_comp_2019_clia_track" /"max3.sl"),
]

def run_command_with_metrics(command, sampling_interval=1, timeout=None):
    start_time = time.time()
    print(f"Running command: {command}")
    process = psutil.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    cpu_usage = []
    memory_usage = []

    try:
        while process.poll() is None:
            try:
                all_cpu = psutil.cpu_percent(interval=sampling_interval, percpu=True)
                cpu_usage.append(all_cpu)

                memory_info = process.memory_info()
                memory_usage.append(memory_info.rss / (1024 * 1024))

            except psutil.NoSuchProcess:
                break

            if timeout and (time.time() - start_time) > timeout:
                process.terminate()
                print(f"Process terminated due to timeout: {command}")
                break

        stdout, stderr = process.communicate()
        end_time = time.time()
        elapsed_time = end_time - start_time

        avg_all_cpu = [sum(core) / len(core) for core in zip(*cpu_usage)] if cpu_usage else []
        avg_cpu_percentage = sum(avg_all_cpu) / len(avg_all_cpu) if avg_all_cpu else 0

        max_memory = max(memory_usage) if memory_usage else 0

        print(f"Command finished: {command}")
        return process.returncode, elapsed_time, stdout, stderr, avg_cpu_percentage, max_memory, avg_all_cpu

    except Exception as e:
        print(f"An error occurred while running the command: {e}")
        return None, None, b"", str(e).encode(), None, None, None


def write_csv_results(filename, results, fieldnames):
    file_exists = Path(filename).exists()
    mode = 'a' if file_exists else 'w'
    with open(filename, mode, newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for result in results:
            writer.writerow(result)

def generate_configs():
    base_config = ConfigManager.load_yaml(py_synth_lab_solver_configs[0])
    configs = []

    for depth in range(3, 10):
        config = base_config.copy()
        config['synthesis_parameters']['max_depth'] = depth
        configs.append(('max_depth', depth, config))

    for complexity in range(3, 10):
        config = base_config.copy()
        config['synthesis_parameters']['max_complexity'] = complexity
        configs.append(('max_complexity', complexity, config))

    for candidates in range(5, 206, 20):
        config = base_config.copy()
        config['synthesis_parameters']['max_candidates_at_each_depth'] = candidates
        configs.append(('max_candidates_at_each_depth', candidates, config))

    base_costs = base_config['synthesis_parameters']['operation_costs']
    cost_multipliers = [0.5, 1.0, 1.5, 2.0]
    for multiplier in cost_multipliers:
        config = base_config.copy()
        config['synthesis_parameters']['operation_costs'] = {op: int(cost * multiplier) for op, cost in base_costs.items()}
        configs.append(('cost_multiplier', multiplier, config))

    return configs

def run_pysynthlab_experiments(timeout_seconds: int = 30):
    run_id = str(uuid.uuid4())
    run_datetime = datetime.now().isoformat()

    fieldnames = [
        "run_id", "run_datetime", "solver", "config", "file", "param_varied", "param_value",
        "return_code", "time", "avg_cpu_percentage", "max_memory_usage", "avg_all_cpu",
        "stdout", "stderr", "candidates_generated", "candidates_pruned", "iterations",
        "time_spent", "solution_height", "solution_complexity", "best_partial_score",
        "grammar_size", "solution_found", "solution_space_coverage", "solver_calls",
        "unique_patterns", "pattern_reuse_ratio",
    ]

    configs = generate_configs()

    for sygus_file in sygus_probs:
        print(f"Processing SyGuS file: {sygus_file}")

        for param_varied, param_value, config in configs:
            results = []
            print(f"Running PySynthLab with {param_varied} = {param_value}")

            config['logging']['file'] = config['logging']['file'].format(
                datetime=datetime.now().strftime("%Y%m%d_%H%M%S"),
                problem=Path(sygus_file).stem,
                root=project_root
            )
            config['logging']['metrics_file'] = config['logging']['metrics_file'].format(
                datetime=datetime.now().strftime("%Y%m%d_%H%M%S"),
                problem=Path(sygus_file).stem,
                root=project_root
            )
            config['synthesis_parameters']['timeout'] = timeout_seconds

            temp_config_file = f"temp_{param_varied}_{param_value}_config.yaml"
            with open(temp_config_file, 'w') as f:
                yaml.dump(config, f)

            metrics_file = config['logging']['metrics_file']
            command = str(project_root) + f"/pysynthlab/bin/python -m src.runner --config {Path.joinpath(Path.cwd(), temp_config_file)} {sygus_file}"
            retcode, elapsed_time, stdout, stderr, avg_cpu_percentage, max_memory, avg_all_cpu = run_command_with_metrics(command, timeout=timeout_seconds)

            max_wait = 5
            wait_time = 0
            while not Path(metrics_file).exists() and wait_time < max_wait:
                time.sleep(0.1)
                wait_time += 0.1

            if Path(metrics_file).exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
            else:
                print(f"Metrics file not found: {metrics_file}")
                metrics = {}

            results.append({
                "run_id": run_id,
                "run_datetime": run_datetime,
                "solver": "PySynthLab",
                "config": temp_config_file,
                "file": str(sygus_file),
                "param_varied": param_varied,
                "param_value": param_value,
                "return_code": retcode,
                "time": elapsed_time,
                "avg_cpu_percentage": avg_cpu_percentage,
                "max_memory_usage": max_memory,
                "avg_all_cpu": json.dumps(avg_all_cpu),
                "stdout": stdout.decode("utf-8"),
                "stderr": stderr.decode("utf-8"),
                **metrics
            })

            Path(temp_config_file).unlink()
            if Path(metrics_file).exists():
                Path(metrics_file).unlink()

            write_csv_results(output_pysynthlab_csv, results, fieldnames)
        print(f"Finished writing results to {output_pysynthlab_csv}")
    print(f"All pysynthlab experiments finished")

if __name__ == "__main__":
    for i in range(3):
        run_pysynthlab_experiments(30)
