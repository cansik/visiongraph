from abc import abstractmethod
from typing import Generic, TypeVar

InputType = TypeVar('InputType')
OutputType = TypeVar('OutputType')


class Processable(Generic[InputType, OutputType]):
    @abstractmethod
    def process(self, data: InputType) -> OutputType:
        pass
