import cv2
import numpy as np
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from visiongraph.recorder.BaseFrameRecorder import BaseFrameRecorder


class MoviePyVideoRecorder(BaseFrameRecorder):
    """
    A video recorder that utilizes MoviePy to write the recorded frames into a video file.
    """

    def __init__(self, width: int, height: int, output_path: str = "video.mp4", fps: float = 30) -> None:
        """
        Initializes the MoviePyVideoRecorder object with the specified parameters.

        :param width: The desired width of the recorded video.
        :param height: The desired height of the recorded video.
        :param output_path: The path to save the recorded video. Defaults to "video.mp4".
        :param fps: The frames per second for the recorded video. Defaults to 30.
        """
        super().__init__()
        self.output_path = output_path
        self.fps = fps
        self.width = width
        self.height = height
        self._images = []

    def open(self) -> None:
        """
        Opens the recorder, resetting the internal image list and calling the superclass's open method.
        """
        self._images = []
        super().open()

    def add_image(self, image: np.ndarray) -> None:
        """
        Adds an image to the recorder's internal image list, converting it from BGR to RGB format before doing so.

        :param image: The image to be added.
        """
        im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        super().add_image(im_rgb)

    def close(self) -> None:
        """
        Closes the recorder, writing the recorded images into a video file using MoviePy and then closing the superclass's closed method.
        """
        clip = ImageSequenceClip(self._images, fps=self.fps)
        clip.write_videofile(self.output_path, logger=None)
        clip.close()
        super().close()
