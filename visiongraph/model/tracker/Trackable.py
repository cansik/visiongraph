from abc import ABC, abstractmethod

from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D


class Trackable(ABC):
    """
    Abstract base class for trackable objects.

    A trackable object is a type of data point that can be assigned to a tracking ID and has a bounding box.
    """

    @property
    @abstractmethod
    def tracking_id(self) -> int:
        """
        Gets the tracking ID assigned to this trackable object.

        :return: The unique identifier for this trackable object.
        """
        pass

    @tracking_id.setter
    @abstractmethod
    def tracking_id(self, value: int):
        """
        Sets the tracking ID for this trackable object.

        :param value: The new tracking ID.
        """
        pass

    @property
    @abstractmethod
    def bounding_box(self) -> BoundingBox2D:
        """
        Gets the 2D bounding box of this trackable object.

        :return: The bounding box containing the location and size information.
        """
        pass
