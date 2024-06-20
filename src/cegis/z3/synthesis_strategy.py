from abc import ABC, abstractmethod


class SynthesisStrategy(ABC):
    @abstractmethod
    def execute_cegis(self):
        pass
