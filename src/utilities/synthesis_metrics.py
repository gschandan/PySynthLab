import json
import time
from pathlib import Path
import psutil
import os


class Metrics:
    def __init__(self, collect_metrics: bool, log_file: str = None):
        self.collect_metrics = collect_metrics
        self.log_file = log_file
        self.candidates_generated = 0
        self.candidates_pruned = 0
        self.iterations = 0
        self.time_spent = 0
        self.solution_height = 0
        self.solution_complexity = 0
        self.partial_satisfaction_scores = []
        self.best_partial_score = 0
        self.grammar_size = 0
        self.solution_found = False
        self.solution_space_explored = 0
        self.total_solution_space = 0
        self.start_time = time.time()
        self.solver_calls = 0
        self.solver_time = 0
        self.unique_patterns_generated = set()
        self.pattern_reuse_count = 0
        self.process = psutil.Process(os.getpid())

        if self.log_file:
            log_dir = Path(self.log_file).parent
            log_dir.mkdir(parents=True, exist_ok=True)

    def update_partial_score(self, score):
        if not self.collect_metrics:
            return
        self.partial_satisfaction_scores.append(score)
        self.best_partial_score = max(self.best_partial_score, score)

    def update_solution_space(self, explored, total):
        if not self.collect_metrics:
            return
        self.solution_space_explored = explored
        self.total_solution_space = total

    def update_solver_metrics(self, time_taken):
        if not self.collect_metrics:
            return
        self.solver_calls += 1
        self.solver_time += time_taken

    def update_pattern_metrics(self, pattern):
        if not self.collect_metrics:
            return
        if pattern in self.unique_patterns_generated:
            self.pattern_reuse_count += 1
        else:
            self.unique_patterns_generated.add(pattern)

    def get_summary(self):
        if not self.collect_metrics:
            return {}
        summary = {
            "candidates_generated": self.candidates_generated,
            "candidates_pruned": self.candidates_pruned,
            "iterations": self.iterations,
            "time_spent": self.time_spent,
            "solution_height": self.solution_height,
            "solution_complexity": self.solution_complexity,
            "best_partial_score": self.best_partial_score,
            "grammar_size": self.grammar_size,
            "solution_found": self.solution_found,
            "solution_space_coverage": self.solution_space_explored / self.total_solution_space if self.total_solution_space else 0,
            "solver_calls": self.solver_calls,
            "unique_patterns": len(self.unique_patterns_generated),
            "pattern_reuse_ratio": self.pattern_reuse_count / self.candidates_generated if self.candidates_generated else 0,
        }
        self.log_metrics(summary)
        return summary

    def log_metrics(self, metrics):
        if self.log_file:
            with open(self.log_file, 'w') as f:
                json.dump(metrics, f, indent=2)

    def __del__(self):
        if self.collect_metrics:
            self.get_summary()