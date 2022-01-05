from abc import ABC, abstractmethod
from argparse import Namespace

from visiongraph.model.parameter.ArgumentConfigurable import ArgumentConfigurable


class GraphNode(ArgumentConfigurable, ABC):
    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def release(self):
        pass

    def configure_and_setup(self, args: Namespace):
        self.configure(args)
        self.setup()
