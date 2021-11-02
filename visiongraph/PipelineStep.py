from abc import ABC, abstractmethod

from visiongraph.ArgumentConfigurable import ArgumentConfigurable


class PipelineStep(ArgumentConfigurable, ABC):
    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def release(self):
        pass

