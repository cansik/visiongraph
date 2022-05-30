from abc import ABC, abstractmethod
from argparse import Namespace, ArgumentParser

import cv2
import numpy as np

from visiongraph import GraphNode


class BaseFrameRecorder(GraphNode[np.ndarray, np.ndarray], ABC):
    def __init__(self):
        self._is_open = False

    def __enter__(self):
        self.open()

    def __exit__(self, type, value, traceback):
        self.close()

    def add_file(self, input_path: str):
        image = cv2.imread(input_path)
        self.add_image(image)

    @abstractmethod
    def open(self):
        self._is_open = True

    @abstractmethod
    def add_image(self, image: np.ndarray):
        pass

    @abstractmethod
    def close(self):
        self._is_open = False

    @property
    def is_open(self):
        return self._is_open

    def setup(self):
        self.open()

    def process(self, data: np.ndarray) -> np.ndarray:
        self.add_image(data)
        return data

    def release(self):
        self.close()

    def configure(self, args: Namespace):
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass
