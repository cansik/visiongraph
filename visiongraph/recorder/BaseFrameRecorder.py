from abc import ABC, abstractmethod

import cv2
import numpy as np


class BaseFrameRecorder(ABC):
    def __enter__(self):
        self.open()

    def __exit__(self, type, value, traceback):
        self.close()

    def add_file(self, input_path: str):
        image = cv2.imread(input_path)
        self.add_image(image)

    @abstractmethod
    def open(self):
        pass

    @abstractmethod
    def add_image(self, image: np.ndarray):
        pass

    @abstractmethod
    def close(self):
        pass