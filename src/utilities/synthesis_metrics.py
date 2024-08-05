class Metrics:
    def __init__(self, collect_metrics: bool):
        self.collect_metrics = collect_metrics
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

    def update_partial_score(self, score):
        if not self.collect_metrics:
            return
        self.partial_satisfaction_scores.append(score)
        self.best_partial_score = max(self.best_partial_score, score)