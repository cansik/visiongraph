from typing import Optional

import cv2
import numpy as np

from visiongraph.recorder.BaseFrameRecorder import BaseFrameRecorder


class CV2VideoRecorder(BaseFrameRecorder):
    def __init__(self, width: int, height: int, output_path: str = "video.mp4", fps: int = 30):
        self.output_path = output_path
        self.fps = fps
        self.width = width
        self.height = height
        self._writer: Optional[cv2.VideoWriter] = None

    def open(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))

    def add_image(self, image: np.ndarray):
        self._writer.write(image)

    def close(self):
        self._writer.release()
