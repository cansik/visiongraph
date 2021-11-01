from abc import ABC, abstractmethod


class PipelineStep(ABC):
    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def release(self):
        pass

    @abstractmethod
    def configure(self, args):
        pass
