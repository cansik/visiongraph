from abc import ABC, abstractmethod
from sys import platform

import cv2
import numpy as np

from visiongraph.GraphNode import GraphNode


class FrameBufferSharingServer(GraphNode[np.ndarray, np.ndarray], ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def send(self, frame: np.array):
        pass

    def process(self, data: np.ndarray) -> np.ndarray:
        rgb_data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        self.send(rgb_data)
        return data

    @staticmethod
    def create(name: str):
        if platform.startswith("darwin"):
            from visiongraph.output.fbs.SyphonServer import SyphonServer
            return SyphonServer(name)
        elif platform.startswith("win"):
            from visiongraph.output.fbs.SpoutServer import SpoutServer
            return SpoutServer(name)
        else:
            raise Exception(f"Platform {platform} is not supported!")
