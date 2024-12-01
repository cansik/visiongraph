from typing import Optional

import cv2
import numpy as np

from visiongraph.recorder.BaseFrameRecorder import BaseFrameRecorder


class CV2VideoRecorder(BaseFrameRecorder):
    def __init__(self, width: Optional[int], height: Optional[int], output_path: str = "video.mp4", fps: float = 30):
        """
        Initializes the CV2VideoRecorder.

        :param width: The video frame width.
        :param height: The video frame height.
        :param output_path: The video file path. Defaults to "video.mp4".
        :param fps: The video frames per second. Defaults to 30.
        """
        super().__init__()
        self.output_path = output_path
        self.fps = fps
        self.width = width
        self.height = height
        self._writer: Optional[cv2.VideoWriter] = None

    def open(self):
        """
        Opens the video recorder.

        If `width` or `height` is provided, initializes the writer.
        Otherwise, calls the parent's `open()` method.
        """
        if self.width is not None or self.height is not None:
            self._init_writer()
        super().open()

    def add_image(self, image: np.ndarray):
        """
        Adds an image to the video frame.

        If `width` or `height` is not provided, calculates it from the image.
        Writes the image to the writer.

        :param image: The input image.
        """
        if self.width is None or self.height is None:
            h, w = image.shape[:2]
            self.width = w
            self.height = h
            self._init_writer()

        self._writer.write(image)

    def close(self):
        """
        Closes the video recorder.

        Releases the writer and calls the parent's `close()` method.
        """
        self._writer.release()
        super().close()

    def _init_writer(self):
        """
        Initializes the video writer.

        Creates a new video writer with the specified fourcc, fps, and frame size.
        """
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))
