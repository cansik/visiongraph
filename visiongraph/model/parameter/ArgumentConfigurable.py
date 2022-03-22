import logging
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

    @staticmethod
    def _get_param(args: Namespace, key: str, default=None):
        if not hasattr(args, key):
            logging.debug(f"Argument {key} has not been parsed, using default value: {default}")
            return default
        return args.__getattribute__(key)
