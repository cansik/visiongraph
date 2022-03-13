from typing import Optional

import cv2
import numpy as np

from visiongraph.recorder.BaseFrameRecorder import BaseFrameRecorder


class CV2VideoRecorder(BaseFrameRecorder):
    def __init__(self, width: Optional[int], height: Optional[int], output_path: str = "video.mp4", fps: int = 30):
        super().__init__()
        self.output_path = output_path
        self.fps = fps
        self.width = width
        self.height = height
        self._writer: Optional[cv2.VideoWriter] = None

    def open(self):
        if self.width is not None or self.height is not None:
            self._init_writer()
        super().open()

    def add_image(self, image: np.ndarray):
        if self.width is None or self.height is None:
            h, w = image.shape[:2]
            self.width = w
            self.height = h
            self._init_writer()

        self._writer.write(image)

    def close(self):
        self._writer.release()
        super().close()

    def _init_writer(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))
