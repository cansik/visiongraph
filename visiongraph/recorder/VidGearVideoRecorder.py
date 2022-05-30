from typing import Optional

import numpy as np
from vidgear.gears import WriteGear

from visiongraph.recorder.BaseFrameRecorder import BaseFrameRecorder


class VidGearVideoRecorder(BaseFrameRecorder):
    def __init__(self, output_path: str = "video.mp4",
                 width: Optional[int] = None, height: Optional[int] = None, fps: float = 30):
        super().__init__()
        self.output_path = output_path
        self.fps = fps
        self.width = width
        self.height = height
        self._writer: Optional[WriteGear] = None

        self.output_params = {
            "-vcodec": "libx264",
            "-pix_fmt": "yuv420p",
            "-crf": 23,
            "-tune": "zerolatency",
            "-input_framerate": self.fps,
            "-disable_force_termination": True
        }

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
        self._writer.close()
        super().close()

    def _init_writer(self):
        self._writer = WriteGear(self.output_path, **self.output_params)
