from argparse import Namespace, ArgumentParser
from typing import Optional, Any

import cv2
import numpy as np
import syphon
from syphon.utils.numpy import copy_image_to_mtl_texture
from syphon.utils.raw import create_mtl_texture

from visiongraph.output.fbs.FrameBufferSharingServer import FrameBufferSharingServer


class SyphonServer(FrameBufferSharingServer):
    def __init__(self, name: str = "SyphonServer"):
        super().__init__(name)

        self.sender: Optional[syphon.SyphonMetalServer] = None
        self.texture: Optional[Any] = None

    def setup(self):
        # setup spout
        self.sender = syphon.SyphonMetalServer(self.name)

    def send(self, frame: np.ndarray, flip_texture: bool = False):
        h, w = frame.shape[:2]

        self._numpy_to_texture(frame, w, h)
        self.sender.publish_frame_texture(self.texture, is_flipped=not flip_texture)

    def release(self):
        self.sender.stop()

    def _numpy_to_texture(self, image: np.ndarray, w: int, h: int):
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)

        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)

        if self.texture is None or self.texture.width() != w or self.texture.height() != h:
            self.texture = create_mtl_texture(self.sender.device, w, h)

        copy_image_to_mtl_texture(image, self.texture)

    def configure(self, args: Namespace):
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass
