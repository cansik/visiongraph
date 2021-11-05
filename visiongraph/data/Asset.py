from abc import ABC, abstractmethod


class Asset(ABC):
    @property
    @abstractmethod
    def exists(self) -> bool:
        pass

    @property
    @abstractmethod
    def path(self) -> str:
        pass

    def prepare(self) -> bool:
        pass
