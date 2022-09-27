import signal
from argparse import ArgumentParser, Namespace
from typing import Callable, Optional

import cv2
import numpy as np

from visiongraph.GraphNode import GraphNode


class ImagePreview(GraphNode[np.ndarray, np.ndarray]):
    def __init__(self, title: str = "Image", wait_time: int = 15,
                 handle_key_callback: Optional[Callable[[int], None]] = None):
        self.title = title
        self.wait_time = wait_time
        self.handle_key_callback = handle_key_callback

    def setup(self):
        pass

    def process(self, data: np.ndarray) -> np.ndarray:
        cv2.imshow(self.title, data)
        key = cv2.waitKey(self.wait_time)

        if self.handle_key_callback is not None and key != 255 and key != -1:
            self.handle_key_callback(key)

        if key & 0xFF == 27:
            signal.raise_signal(signal.SIGINT)

        return data

    def release(self):
        pass

    def configure(self, args: Namespace):
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass
