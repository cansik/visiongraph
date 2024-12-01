import time
from threading import Thread

import numpy as np

from visiongraph.recorder.FrameSetRecorder import FrameSetRecorder


class AsyncFrameSetRecorder(FrameSetRecorder):
    """
    A class to record frames asynchronously.

    This class is a subclass of FrameSetRecorder and provides asynchronous recording.
    """

    def __init__(self, output_path: str = "recordings"):
        """
        Initializes the AsyncFrameSetRecorder object with an output path.

        :param output_path: The directory where recorded frames will be saved. Defaults to "recordings".
        """
        super().__init__(output_path)
        self._running = True

        # Create a new thread for writer loop
        self._writer_thread = Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

        # Initialize image index
        self._image_index = 0

    def open(self):
        """
        Opens the recording by resetting the image index.
        """
        self._image_index = 0
        super().open()

    def add_image(self, image: np.ndarray):
        """
        Adds an image to the recording.

        :param image: The image to be added.
        """
        super().add_image(image)

    def close(self):
        """
        Closes the recording by removing frames from the queue and waiting for all images to be written.
        """
        while not self._frames.empty():
            time.sleep(0.1)

    def shutdown(self):
        """
        Shuts down the recorder by closing it and stopping the writer thread.
        """
        self.close()
        self._running = False

    def _writer_loop(self):
        """
        The writer loop that continuously writes images to the output path.

        This loop runs in a separate thread and checks if there are frames available to write.
        If yes, it gets an image from the queue, writes it, and increments the image index.
        """
        while self._running or not self._frames.empty():
            image = self._frames.get()
            self._write_image(self._image_index, image)
            self._image_index += 1
