import os
from queue import Queue

import cv2
import numpy as np

from visiongraph.recorder.BaseFrameRecorder import BaseFrameRecorder


class FrameSetRecorder(BaseFrameRecorder):
    def __init__(self, output_path: str = "recordings"):
        super().__init__()
        self.output_path = output_path
        self._frames = Queue()

    def open(self):
        # clear all items
        self.clear()
        os.makedirs(self.output_path, exist_ok=True)
        super().open()

    def add_image(self, image: np.ndarray):
        self._frames.put(image)

    def close(self):
        i = 0
        while not self._frames.empty():
            image = self._frames.get()
            self._write_image(i, image)
            i += 1
        super().close()

    def clear(self):
        while self._frames.qsize():
            self._frames.get_nowait()

    def _write_image(self, id: int, image: np.ndarray):
        output_path = os.path.join(self.output_path, f"{id:06d}.png")
        cv2.imwrite(output_path, image)
