import signal
from argparse import ArgumentParser, Namespace

import cv2
import numpy as np

from visiongraph.GraphNode import GraphNode


class ImagePreview(GraphNode[np.ndarray, np.ndarray]):
    def __init__(self, title: str = "Image", wait_time: int = 15):
        self.title = title
        self.wait_time = wait_time

    def setup(self):
        pass

    def process(self, data: np.ndarray) -> np.ndarray:
        cv2.imshow(self.title, data)
        if cv2.waitKey(self.wait_time) & 0xFF == 27:
            signal.raise_signal(signal.SIGINT)

        return data

    def release(self):
        pass

    def configure(self, args: Namespace):
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass
