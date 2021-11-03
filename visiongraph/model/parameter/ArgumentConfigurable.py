from abc import abstractmethod, ABC
from argparse import ArgumentParser, Namespace


class ArgumentConfigurable(ABC):
    @abstractmethod
    def configure(self, args: Namespace):
        pass

    @staticmethod
    @abstractmethod
    def add_params(parser: ArgumentParser):
        pass
