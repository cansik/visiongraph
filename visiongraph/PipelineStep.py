from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace


class PipelineStep(ABC):
    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def release(self):
        pass

    @abstractmethod
    def configure(self, args: Namespace):
        pass

    @staticmethod
    @abstractmethod
    def add_params(parser: ArgumentParser):
        pass
