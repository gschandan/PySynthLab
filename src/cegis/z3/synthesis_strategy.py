from abc import ABC, abstractmethod


class SynthesisStrategy(ABC):
    def __init__(self):
        self.solution_found = False

    @abstractmethod
    def execute_cegis(self):
        pass

    def set_solution_found(self):
        self.solution_found = True

    def get_solution_found(self):
        return self.solution_found
