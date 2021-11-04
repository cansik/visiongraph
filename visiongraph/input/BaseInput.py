from abc import abstractmethod, ABC
from argparse import ArgumentParser, Namespace, ArgumentError
from typing import Optional

import numpy as np
import cv2

from visiongraph.PipelineNode import PipelineNode
from visiongraph.model.parameter.NamedParameter import RotationParameter, FlipParameter
from visiongraph.util.ArgUtils import add_dict_choice_argument


class BaseInput(PipelineNode, ABC):
    @abstractmethod
    def __init__(self):
        self.width = 640
        self.height = 480
        self.fps = 30
        self.rotate: Optional[int] = None
        self.flip: Optional[int] = None

    @abstractmethod
    def read(self) -> (int, Optional[np.ndarray]):
        pass

    def _post_process(self, ts: int, image: Optional[np.ndarray]) -> (int, Optional[np.ndarray]):
        if image is None:
            return ts, image

        if self.rotate is not None:
            image = cv2.rotate(image, self.rotate)

        if self.flip is not None:
            image = cv2.flip(image, self.flip)

        return ts, image

    @abstractmethod
    def configure(self, args: Namespace):
        self.width, self.height = args.input_size
        self.fps = args.input_fps
        self.rotate = args.input_rotate
        self.flip = args.input_flip

    @staticmethod
    @abstractmethod
    def add_params(parser: ArgumentParser):
        try:
            parser.add_argument("--input-size", default=[640, 480], type=int, nargs=2, metavar=('width', 'height'),
                                help="Requested input media size.")
            parser.add_argument("--input-fps", default=30, type=int, help="Requested input media framerate.")
            add_dict_choice_argument(parser, RotationParameter, "--input-rotate", help="Rotate input media",
                                     default=None)
            add_dict_choice_argument(parser, FlipParameter, "--input-flip", help="Flip input media", default=None)
        except ArgumentError as ex:
            if ex.message.startswith("conflicting"):
                return
            raise ex
