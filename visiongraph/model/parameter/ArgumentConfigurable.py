import logging
from abc import abstractmethod, ABC
from argparse import ArgumentParser, Namespace


class ArgumentConfigurable(ABC):
    """
    Abstract base class for classes that can be configured from command-line arguments.
    """

    @abstractmethod
    def configure(self, args: Namespace):
        """
        Configures the object with values parsed from command-line arguments.

        :param args: The parsed command-line argument namespace.
        """
        pass

    @staticmethod
    @abstractmethod
    def add_params(parser: ArgumentParser):
        """
        Adds command-line parameters to a parser instance.

        :param parser: The parser instance to be extended.
        """
        pass

    @staticmethod
    def _get_param(args: Namespace, key: str, default=None) -> any:
        """
        Retrieves the value of a parsed argument with a fallback to a default value if not found.

        :param args: The parsed command-line argument namespace.
        :param key: The name of the argument to retrieve.
        :param default: The default value to return if the argument is not found. Defaults to None.

        :return: The retrieved argument value or its default value if not found.
        """
        if not hasattr(args, key):
            logging.debug(f"Argument {key} has not been parsed, using default value: {default}")
            return default
        return args.__getattribute__(key)
