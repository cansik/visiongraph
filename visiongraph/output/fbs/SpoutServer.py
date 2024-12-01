import logging
from argparse import ArgumentParser, Namespace
from typing import Optional

import SpoutGL
import cv2
import numpy as np
from OpenGL import GL

from visiongraph.output.fbs.FrameBufferSharingServer import FrameBufferSharingServer


class SpoutServer(FrameBufferSharingServer):
    """
    A class to represent a Spout server with shared frame buffer capabilities.
    """

    def __init__(self, name: str = "SpoutServer"):
        """
        Initializes the SpoutServer object with a default name and sets up the context.

        :param name: The name of the Spout server. Defaults to "SpoutServer".
        """
        super().__init__(name)
        self.ctx: Optional[SpoutGL.SpoutSender] = None

    def setup(self):
        """
        Sets up the Spout sender and sets its name.
        """
        # setup spout
        self.ctx = SpoutGL.SpoutSender()
        self.ctx.setSenderName(self.name)

    def send(self, frame: np.ndarray, send_alpha: bool = True, flip_texture: bool = False):
        """
        Sends a frame to the Spout receiver.

        :param frame: The frame to be sent.
        :param send_alpha: Whether to send an alpha channel. Defaults to True.
        :param flip_texture: Whether to flip the texture. Defaults to False.

        :return: Whether the image was sent successfully.
        """
        h, w = frame.shape[:2]

        if send_alpha and frame.shape[2] < 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)

        success = self.ctx.sendImage(frame, w, h, GL.GL_RGBA, flip_texture, 0)

        # This fixes the CPU receiver (first frame is discarded)
        # More information: https://github.com/jlai/Python-SpoutGL/issues/15
        self.ctx.setCPUshare(True)

        if not success:
            logging.warning("Could not send spout image.")
            return

        # Indicate that a frame is ready to read
        self.ctx.setFrameSync(self.name)

    def release(self):
        """
        Releases the Spout sender.
        """
        self.ctx.releaseSender()

    def configure(self, args: Namespace):
        """
        Configures the Spout server based on the provided arguments.

        :param args: The parsed command-line arguments.
        """
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        """
        Adds parameters to the argument parser for the Spout server.

        :param parser: The argument parser to add parameters to.
        """
        pass
