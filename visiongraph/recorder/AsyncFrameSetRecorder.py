import time
from threading import Thread

import numpy as np

from visiongraph.recorder.FrameSetRecorder import FrameSetRecorder


class AsyncFrameSetRecorder(FrameSetRecorder):
    def __init__(self, output_path: str = "recordings"):
        super().__init__(output_path)
        self._running = True

        self._writer_thread = Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

        self._image_index = 0

    def open(self):
        self._image_index = 0
        super().open()

    def add_image(self, image: np.ndarray):
        super().add_image(image)

    def close(self):
        while not self._frames.empty():
            time.sleep(0.1)

    def shutdown(self):
        self.close()
        self._running = False

    def _writer_loop(self):
        while self._running or not self._frames.empty():
            image = self._frames.get()
            self._write_image(self._image_index, image)
            self._image_index += 1
