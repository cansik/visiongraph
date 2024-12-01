from typing import Optional

import numpy as np
from vidgear.gears import WriteGear

from visiongraph.recorder.BaseFrameRecorder import BaseFrameRecorder


class VidGearVideoRecorder(BaseFrameRecorder):
    def __init__(self, output_path: str = "video.mp4",
                 width: Optional[int] = None, height: Optional[int] = None, fps: float = 30):
        """
        Initializes the VidGearVideoRecorder.

        :param output_path: The path where the video will be saved. Defaults to "video.mp4".
        :param width: The width of the video in pixels. Defaults to None.
        :param height: The height of the video in pixels. Defaults to None.
        :param fps: The frames per second for the video. Defaults to 30.

        """
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
        """
        Opens the video writer and initializes the recorder.

        If width or height is provided, it will create a WriteGear object.
        Otherwise, it will only call the superclass's open method.
        """
        if self.width is not None or self.height is not None:
            self._init_writer()
        super().open()

    def add_image(self, image: np.ndarray):
        """
        Adds an image to the video.

        If width or height is not provided, it will automatically detect them from the image and create a WriteGear object.
        Otherwise, it will write the image using the existing writer.

        :param image: The image to be added to the video.
        """
        if self.width is None or self.height is None:
            h, w = image.shape[:2]
            self.width = w
            self.height = h
            self._init_writer()

        self._writer.write(image)

    def close(self):
        """
        Closes the video writer and recorder.

        This method calls the superclass's close method as well.
        """
        self._writer.close()
        super().close()

    def _init_writer(self):
        """
        Initializes the WriteGear object with the specified output path and parameters.

        If width or height is provided in the __init__ method, it will be used to create a WriteGear object.
        Otherwise, this method is called automatically when add_image is called with an image that has a different size than the previously recorded images.
        """
        self._writer = WriteGear(self.output_path, **self.output_params)
