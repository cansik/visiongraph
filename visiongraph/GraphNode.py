from abc import ABC, abstractmethod
from argparse import Namespace
from typing import TypeVar

from visiongraph.Processable import Processable
from visiongraph.model.parameter.ArgumentConfigurable import ArgumentConfigurable

InputType = TypeVar('InputType')
OutputType = TypeVar('OutputType')


class GraphNode(Processable[InputType, OutputType], ArgumentConfigurable, ABC):
    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def process(self, data: InputType) -> OutputType:
        pass

    @abstractmethod
    def release(self):
        pass

    def configure_and_setup(self, args: Namespace):
        self.configure(args)
        self.setup()
