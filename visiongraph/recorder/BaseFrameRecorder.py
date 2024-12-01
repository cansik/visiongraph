from abc import ABC, abstractmethod
from argparse import Namespace, ArgumentParser

import cv2
import numpy as np

from visiongraph.GraphNode import GraphNode


class BaseFrameRecorder(GraphNode[np.ndarray, np.ndarray], ABC):
    """
    Abstract base class for recording frames, extending the GraphNode functionality.
    """

    def __init__(self):
        """
        Initializes the BaseFrameRecorder object and sets the open state to False.
        """
        self._is_open = False

    def __enter__(self):
        """
        Opens the frame recorder upon entering the context.
        """
        self.open()

    def __exit__(self, type, value, traceback):
        """
        Closes the frame recorder upon exiting the context.

        :param type: The exception type, if any.
        :param value: The exception value, if any.
        :param traceback: The traceback object, if any.
        """
        self.close()

    def add_file(self, input_path: str):
        """
        Reads an image from a file and adds it to the recorder.

        :param input_path: The file path to the image.
        """
        image = cv2.imread(input_path)
        self.add_image(image)

    @abstractmethod
    def open(self):
        """
        Opens the frame recorder for recording frames.
        """
        self._is_open = True

    @abstractmethod
    def add_image(self, image: np.ndarray):
        """
        Adds an image to the recorder.

        :param image: The image to be added.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Closes the frame recorder and releases resources.
        """
        self._is_open = False

    @property
    def is_open(self):
        """
        Indicates whether the frame recorder is currently open.

        :return: True if the recorder is open, False otherwise.
        """
        return self._is_open

    def setup(self):
        """
        Prepares the recorder for operation by opening it.
        """
        self.open()

    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Processes the input data by adding it as an image to the recorder.

        :param data: The input data to be processed.

        :return: The processed input data.
        """
        self.add_image(data)
        return data

    def release(self):
        """
        Releases the recorder resources by closing it.
        """
        self.close()

    def configure(self, args: Namespace):
        """
        Configures the recorder with command-line arguments.

        :param args: The parsed command-line arguments.
        """
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        """
        Adds command-line parameters specific to the frame recorder.

        :param parser: The argument parser to add parameters to.
        """
        pass
