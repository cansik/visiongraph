from typing import Union, Sequence

import numpy as np


class Size2D:
    """
    A class to represent a two-dimensional size with width and height.
    """

    def __init__(self, width: float, height: float):
        """
        Initializes the Size2D object with given width and height.

        :param width: The width of the size.
        :param height: The height of the size.
        """
        self.width = width
        self.height = height

    def __iter__(self):
        """
        Iterates over the width and height of the size.

        Yields:
            float: The width and height in succession.
        """
        yield self.width
        yield self.height

    def scale(self, width: float, height: float) -> "Size2D":
        """
        Scales the width and height by given factors.

        :param width: The factor by which to scale the width.
        :param height: The factor by which to scale the height.

        :return: A new Size2D object with scaled dimensions.
        """
        return Size2D(
            self.width * width,
            self.height * height)

    @staticmethod
    def from_array(data: Union[Sequence, np.ndarray]):
        """
        Creates a Size2D object from a sequence or numpy array.

        :param data: A sequence or numpy array containing at least two elements.

        :return: A Size2D object initialized with the first two elements of the data.
        """
        if isinstance(data, np.ndarray):
            data = data.flat

        return Size2D(data[0], data[1])

    @staticmethod
    def from_image(image: np.ndarray):
        """
        Creates a Size2D object from an image's dimensions.

        :param image: A numpy array representing the image.

        :return: A Size2D object with width and height corresponding to the image dimensions.
        """
        h, w = image.shape[:2]
        return Size2D(float(w), float(h))

    def __repr__(self):
        """
        Returns a string representation of the Size2D object.

        :return: A formatted string describing the width and height.
        """
        return f"Size2D(w={self.width:.4f}, h={self.height:.4f})"
