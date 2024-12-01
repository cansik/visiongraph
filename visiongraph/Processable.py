from abc import abstractmethod
from typing import Generic, TypeVar

InputType = TypeVar('InputType')
OutputType = TypeVar('OutputType')


class Processable(Generic[InputType, OutputType]):
    """
    An abstract base class for objects that can process input data.
    """

    @abstractmethod
    def process(self, data: InputType) -> OutputType:
        """
        Processes the given input data and returns the result.

        :param data: The input data to be processed.

        :return: The processed output.
        """
        pass
