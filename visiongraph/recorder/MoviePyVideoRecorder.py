from typing import Optional

import cv2
import numpy as np

from visiongraph.recorder.BaseFrameRecorder import BaseFrameRecorder
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip


class MoviePyVideoRecorder(BaseFrameRecorder):
    def __init__(self, width: int, height: int, output_path: str = "video.mp4", fps: float = 30):
        super().__init__()
        self.output_path = output_path
        self.fps = fps
        self.width = width
        self.height = height
        self._images = []

    def open(self):
        self._images = []
        super().open()

    def add_image(self, image: np.ndarray):
        im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        super().add_image(im_rgb)

    def close(self):
        clip = ImageSequenceClip(self._images, fps=self.fps)
        clip.write_videofile(self.output_path, logger=None)
        clip.close()
        super().close()
