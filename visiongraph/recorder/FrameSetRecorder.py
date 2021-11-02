import os

import cv2
import numpy as np

from visiongraph.recorder.BaseFrameRecorder import BaseFrameRecorder


class FrameSetRecorder(BaseFrameRecorder):
    def __init__(self, output_path: str = "recordings"):
        self.output_path = output_path
        self._frames = []

    def open(self):
        self._frames = []
        os.makedirs(self.output_path, exist_ok=True)

    def add_image(self, image: np.ndarray):
        self._frames.append(image)

    def close(self):
        for i, image in enumerate(self._frames):
            output_path = os.path.join(self.output_path, f"{i:04d}.png")
            cv2.imwrite(output_path, image)
