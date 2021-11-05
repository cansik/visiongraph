from abc import ABC, abstractmethod

from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D


class Trackable(ABC):
    @property
    @abstractmethod
    def tracking_id(self) -> int:
        pass

    @tracking_id.setter
    @abstractmethod
    def tracking_id(self, value: int):
        pass

    @property
    @abstractmethod
    def bounding_box(self) -> BoundingBox2D:
        pass
