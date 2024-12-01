from argparse import Namespace, ArgumentParser
from typing import Optional, Any

import cv2
import numpy as np
import syphon
from syphon.utils.numpy import copy_image_to_mtl_texture
from syphon.utils.raw import create_mtl_texture

from visiongraph.output.fbs.FrameBufferSharingServer import FrameBufferSharingServer


class SyphonServer(FrameBufferSharingServer):
    """
    A class to represent a Syphon server for sharing frames.
    """

    def __init__(self, name: str = "SyphonServer"):
        """
        Initializes the SyphonServer object.

        :param name: The name of the server. Defaults to "SyphonServer".
        """
        super().__init__(name)

        self.sender: Optional[syphon.SyphonMetalServer] = None
        self.texture: Optional[Any] = None

    def setup(self):
        """
        Sets up the spout object for publishing.
        """
        # setup spout
        self.sender = syphon.SyphonMetalServer(self.name)

    def send(self, frame: np.ndarray, flip_texture: bool = False):
        """
        Publishes a frame to the Syphon server.

        :param frame: The frame to be published.
        :param flip_texture: Whether to flip the texture. Defaults to False.
        """
        h, w = frame.shape[:2]

        self._numpy_to_texture(frame, w, h)
        self.sender.publish_frame_texture(self.texture, is_flipped=not flip_texture)

    def release(self):
        """
        Stops the spout object.
        """
        self.sender.stop()

    def _numpy_to_texture(self, image: np.ndarray, w: int, h: int) -> None:
        """
        Converts a numpy array to a texture.

        :param image: The numpy array to be converted.
        :param w: The width of the image.
        :param h: The height of the image.
        """
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)

        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)

        if self.texture is None or self.texture.width() != w or self.texture.height() != h:
            self.texture = create_mtl_texture(self.sender.device, w, h)

        copy_image_to_mtl_texture(image, self.texture)

    def configure(self, args: Namespace) -> None:
        """
        Configures the SyphonServer object based on the provided arguments.

        :param args: The namespace containing configuration options.
        """
        pass

    @staticmethod
    def add_params(parser: ArgumentParser) -> None:
        """
        Adds parameters to the ArgumentParser for configuration.

        :param parser: The parser to be updated.
        """
        pass
