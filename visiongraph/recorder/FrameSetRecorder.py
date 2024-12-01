import os
from queue import Queue

import cv2
import numpy as np

from visiongraph.recorder.BaseFrameRecorder import BaseFrameRecorder


class FrameSetRecorder(BaseFrameRecorder):
    """
    A recorder for handling and saving a set of frames as images.
    """

    def __init__(self, output_path: str = "recordings"):
        """
        Initializes the FrameSetRecorder with a specified output path.

        :param output_path: The path to the directory for storing recordings. Default is "recordings".
        """
        super().__init__()
        self.output_path = output_path
        self._frames = Queue()

    def open(self):
        """
        Prepares the recorder for operation by clearing existing frames and creating the output directory.
        """
        self.clear()
        os.makedirs(self.output_path, exist_ok=True)
        super().open()

    def add_image(self, image: np.ndarray):
        """
        Adds an image to the frame queue for later saving.

        :param image: The image to be added to the queue.
        """
        self._frames.put(image)

    def close(self):
        """
        Closes the recorder and writes all queued frames to disk.
        """
        i = 0
        while not self._frames.empty():
            image = self._frames.get()
            self._write_image(i, image)
            i += 1
        super().close()

    def clear(self):
        """
        Clears all frames from the queue.
        """
        while self._frames.qsize():
            self._frames.get_nowait()

    def _write_image(self, id: int, image: np.ndarray):
        """
        Writes a single image to the output directory with a formatted filename.

        :param id: The identifier for the image to be saved.
        :param image: The image to be saved.
        """
        output_path = os.path.join(self.output_path, f"{id:06d}.png")
        cv2.imwrite(output_path, image)
