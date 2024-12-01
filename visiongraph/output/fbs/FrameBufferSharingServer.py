from abc import ABC, abstractmethod
from sys import platform

import cv2
import numpy as np

from visiongraph.GraphNode import GraphNode


class FrameBufferSharingServer(GraphNode[np.ndarray, np.ndarray], ABC):
    """
    Abstract base class for frame buffer sharing servers.

    This class provides a basic interface for sending and processing RGB frames.
    Subclasses should implement the send method to provide actual server functionality.
    """

    def __init__(self, name: str):
        """
        Initializes the FrameBufferSharingServer object.

        :param name: The name of the frame buffer sharing server.
        """
        self.name = name

    @abstractmethod
    def send(self, frame: np.ndarray, flip_texture: bool = False) -> None:
        """
        Sends the provided frame to a client.

        :param frame: The RGB frame to be sent.
        :param flip_texture: Whether to flip the texture. Defaults to False.
        """
        pass

    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Processes the provided frame and sends it to a client.

        First, converts the BGR frame to RGB using OpenCV's cv2.cvtColor function,
        then calls the send method on this instance with the converted frame.
        Finally, returns the original BGR frame unchanged.

        :param data: The input BGR frame.

        :return: The original BGR frame.
        """
        rgb_data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        self.send(rgb_data)
        return data

    @staticmethod
    def create(name: str) -> "FrameBufferSharingServer":
        """
        Creates a new instance of the FrameBufferSharingServer class.

        Depending on the platform, it returns either a SyphonServer or SpoutServer instance.
        If the platform is not supported, it raises an exception with a descriptive message.

        :param name: The name of the frame buffer sharing server.

        :return: A new FrameBufferSharingServer instance.
        """
        if platform.startswith("darwin"):
            from visiongraph.output.fbs.SyphonServer import SyphonServer
            return SyphonServer(name)
        elif platform.startswith("win"):
            from visiongraph.output.fbs.SpoutServer import SpoutServer
            return SpoutServer(name)
        else:
            raise Exception(f"Platform {platform} is not supported!")
