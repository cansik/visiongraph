from typing import Sequence, Union

import numpy as np
import vector

from visiongraph.model.geometry.Size2D import Size2D
from visiongraph.util import MathUtils


class BoundingBox2D:
    """
    A class to represent a 2-dimensional bounding box.
    """

    def __init__(self, x_min: float, y_min: float, width: float, height: float):
        """
        Initializes a BoundingBox2D object with the given coordinates and dimensions.

        Args:
            x_min (float): The minimum x-coordinate of the bounding box.
            y_min (float): The minimum y-coordinate of the bounding box.
            width (float): The width of the bounding box.
            height (float): The height of the bounding box.
        """
        self.x_min = x_min
        self.y_min = y_min
        self.width = width
        self.height = height

    def __iter__(self):
        """
        Iterates over the bounding box properties.

        Yields:
            float: The minimum x-coordinate, minimum y-coordinate, width, and height.
        """
        yield self.x_min
        yield self.y_min
        yield self.width
        yield self.height

    @property
    def center(self) -> vector.Vector2D:
        """
        Calculates the center point of the bounding box.

        Returns:
            vector.Vector2D: The center point of the bounding box.
        """
        return vector.obj(x=self.x_min + self.width * 0.5, y=self.y_min + self.height * 0.5)

    @property
    def top_left(self) -> vector.Vector2D:
        """
        Gets the top-left corner of the bounding box.

        Returns:
            vector.Vector2D: The top-left corner of the bounding box.
        """
        return vector.obj(x=self.x_min, y=self.y_min)

    @property
    def top_right(self) -> vector.Vector2D:
        """
        Gets the top-right corner of the bounding box.

        Returns:
            vector.Vector2D: The top-right corner of the bounding box.
        """
        return vector.obj(x=self.x_min + self.width, y=self.y_min)

    @property
    def bottom_right(self) -> vector.Vector2D:
        """
        Gets the bottom-right corner of the bounding box.

        Returns:
            vector.Vector2D: The bottom-right corner of the bounding box.
        """
        return vector.obj(x=self.x_min + self.width, y=self.y_min + self.height)

    @property
    def bottom_left(self) -> vector.Vector2D:
        """
        Gets the bottom-left corner of the bounding box.

        Returns:
            vector.Vector2D: The bottom-left corner of the bounding box.
        """
        return vector.obj(x=self.x_min, y=self.y_min + self.height)

    @property
    def x_max(self) -> float:
        """
        Computes the maximum x-coordinate of the bounding box.

        Returns:
            float: The maximum x-coordinate of the bounding box.
        """
        return self.x_min + self.width

    @property
    def y_max(self):
        """
        Computes the maximum y-coordinate of the bounding box.

        Returns:
            float: The maximum y-coordinate of the bounding box.
        """
        return self.y_min + self.height

    @property
    def size(self) -> Size2D:
        """
        Gets the size of the bounding box as a Size2D object.

        Returns:
            Size2D: The size of the bounding box.
        """
        return Size2D(self.width, self.height)

    def to_array(self, tl_br_format: bool = False) -> np.ndarray:
        """
        Converts the bounding box to a NumPy array representation.

        Args:
            tl_br_format (bool): If True, returns the top-left and bottom-right format.

        Returns:
            np.ndarray: Array representation of the bounding box.
        """
        if tl_br_format:
            return np.array([self.x_min, self.y_min, self.x_min + self.width, self.y_min + self.height])
        return np.array([self.x_min, self.y_min, self.width, self.height])

    def scale(self, width: float, height: float) -> "BoundingBox2D":
        """
        Scales the bounding box by the specified width and height factors.

        Args:
            width (float): The factor by which to scale the width.
            height (float): The factor by which to scale the height.

        Returns:
            BoundingBox2D: A new BoundingBox2D instance with scaled dimensions.
        """
        return BoundingBox2D(
            self.x_min * width,
            self.y_min * height,
            self.width * width,
            self.height * height)

    def scale_with(self, size: Size2D) -> "BoundingBox2D":
        """
        Scales the bounding box with the given Size2D object.

        Args:
            size (Size2D): The size object to scale the bounding box.

        Returns:
            BoundingBox2D: A new BoundingBox2D instance with scaled dimensions.
        """
        return self.scale(size.width, size.height)

    def scale_centered(self, width: float, height: float) -> "BoundingBox2D":
        """
        Scales the bounding box centered on its current center.

        Args:
            width (float): Scaling factor for the width.
            height (float): Scaling factor for the height.

        Returns:
            BoundingBox2D: A new BoundingBox2D instance with centered scaling.
        """
        dx = self.width * width
        dy = self.height * height
        return self.add_border(dx, dy)

    def add_border(self, dx: float, dy: float) -> "BoundingBox2D":
        """
        Adds a border of specified width and height to the bounding box.

        Args:
            dx (float): Width of the border to add.
            dy (float): Height of the border to add.

        Returns:
            BoundingBox2D: A new BoundingBox2D instance with added borders.
        """
        return BoundingBox2D(
            self.x_min - (dx * 0.5),
            self.y_min - (dy * 0.5),
            self.width + dx,
            self.height + dy)

    def add_border_with(self, size: Size2D) -> "BoundingBox2D":
        """
        Adds a border using the specified Size2D object.

        Args:
            size (Size2D): Size object containing the width and height of the border.

        Returns:
            BoundingBox2D: A new BoundingBox2D instance with added borders.
        """
        return self.add_border(size.width, size.height)

    @staticmethod
    def from_array(data: Union[Sequence, np.ndarray], tl_br_format: bool = False):
        """
        Creates a BoundingBox2D from a sequence or NumPy array.

        Args:
            data (Union[Sequence, np.ndarray]): The data representing the bounding box.
            tl_br_format (bool): If True, the data is interpreted in top-left and bottom-right format.

        Returns:
            BoundingBox2D: A new BoundingBox2D instance constructed from the provided data.
        """
        if isinstance(data, np.ndarray):
            data = data.flat

        if tl_br_format:
            return BoundingBox2D(data[0], data[1], data[2] - data[0], data[3] - data[1])

        return BoundingBox2D(data[0], data[1], data[2], data[3])

    @staticmethod
    def from_image(image: np.ndarray):
        """
        Creates a BoundingBox2D that encompasses the entire image.

        Args:
            image (np.ndarray): The image to create the bounding box from.

        Returns:
            BoundingBox2D: A new BoundingBox2D instance representing the full image size.
        """
        h, w = image.shape[:2]
        return BoundingBox2D(0, 0, float(w), float(h))

    @staticmethod
    def from_kernel(x: int, y: int, kernel_size: int):
        """
        Creates a BoundingBox2D from a kernel position and size.

        Args:
            x (int): The x-coordinate of the kernel.
            y (int): The y-coordinate of the kernel.
            kernel_size (int): The size of the kernel.

        Returns:
            BoundingBox2D: A new BoundingBox2D instance representing the kernel.
        """
        shift = kernel_size // 2
        return BoundingBox2D(x - shift, y - shift, kernel_size, kernel_size)

    def intersection_over_union(self, box: "BoundingBox2D", epsilon: float = 1e-5) -> float:
        """
        Computes the Intersection over Union (IoU) with another bounding box.

        Args:
            box (BoundingBox2D): The bounding box to compute the IoU with.
            epsilon (float): A small value to avoid division by zero.

        Returns:
            float: The IoU value between the two bounding boxes.
        """
        return MathUtils.intersection_over_union(self.to_array(True), box.to_array(True), epsilon)

    def contains(self, p: vector.Vector2D) -> bool:
        """
        Checks whether a point is inside the bounding box.

        Args:
            p (vector.Vector2D): The point to check.

        Returns:
            bool: True if the point is inside the bounding box, False otherwise.
        """
        if self.x_min < p.x < self.x_max:
            if self.y_min < p.y < self.y_max:
                return True
        return False

    def __repr__(self):
        """
        Returns a string representation of the bounding box.

        Returns:
            str: A string describing the bounding box with its coordinates and dimensions.
        """
        return f"BoundingBox2D(x={self.x_min:.4f}, y={self.y_min:.4f}, w={self.width:.4f}, h={self.height:.4f})"
