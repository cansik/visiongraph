from abc import ABC, abstractmethod
from argparse import ArgumentParser


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

    @staticmethod
    @abstractmethod
    def add_params(parser: ArgumentParser):
        pass
