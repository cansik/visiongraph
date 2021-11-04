from abc import ABC, abstractmethod

from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D


class Trackable(ABC):
    def __init__(self, tracking_id: int):
        self.tracking_id = tracking_id

    @property
    @abstractmethod
    def bounding_box(self) -> BoundingBox2D:
        pass
