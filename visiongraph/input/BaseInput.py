from abc import abstractmethod, ABC
from argparse import ArgumentParser, Namespace, ArgumentError
from typing import Optional

import numpy as np
import cv2

from visiongraph.GraphNode import GraphNode, InputType, OutputType
from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.model.parameter.NamedParameter import RotationParameter, FlipParameter
from visiongraph.util import ImageUtils
from visiongraph.util.ArgUtils import add_dict_choice_argument


class BaseInput(GraphNode[None, np.ndarray], ABC):
    @abstractmethod
    def __init__(self):
        self.width = 640
        self.height = 480
        self.fps = 30

        self.rotate: Optional[int] = None
        self.flip: Optional[int] = None
        self.crop: Optional[BoundingBox2D] = None
        self.mask: Optional[np.ndarray] = None

        self.raw_input = False

    @abstractmethod
    def read(self) -> (int, Optional[np.ndarray]):
        pass

    def process(self, data: InputType) -> OutputType:
        ts, image = self.read()
        return image

    def _post_process(self, ts: int, image: Optional[np.ndarray]) -> (int, Optional[np.ndarray]):
        if image is None:
            return ts, image

        if self.rotate is not None:
            image = cv2.rotate(image, self.rotate)

        if self.flip is not None:
            image = cv2.flip(image, self.flip)

        if self.mask is not None:
            image = ImageUtils.apply_mask(image, self.mask)

        if self.crop is not None:
            image = ImageUtils.roi(image, self.crop)

            if 0 in image.shape[:2]:
                return ts, None

        # prepare image to be 3 channel
        if not self.raw_input and (len(image.shape) < 3 or image.shape[2] == 1):
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        return ts, image

    @abstractmethod
    def configure(self, args: Namespace):
        self.width, self.height = args.input_size
        self.fps = args.input_fps
        self.rotate = args.input_rotate
        self.flip = args.input_flip
        self.raw_input = args.raw_input

        if args.input_mask is not None:
            self.mask = cv2.imread(args.input_mask, cv2.IMREAD_GRAYSCALE)

            if self.mask.shape[0] != self.height or self.mask.shape[1] != self.width:
                self.mask = cv2.resize(self.mask, (self.width, self.height))

        if args.input_crop is not None:
            self.crop = BoundingBox2D.from_array(args.input_crop)

    @staticmethod
    @abstractmethod
    def add_params(parser: ArgumentParser):
        try:
            parser.add_argument("--input-size", default=[640, 480], type=int, nargs=2, metavar=("width", "height"),
                                help="Requested input media size.")
            parser.add_argument("--input-fps", default=30, type=int, help="Requested input media framerate.")
            add_dict_choice_argument(parser, RotationParameter, "--input-rotate", help="Rotate input media",
                                     default=None)
            add_dict_choice_argument(parser, FlipParameter, "--input-flip", help="Flip input media", default=None)
            parser.add_argument("--input-mask", default=None, type=str, help="Path to the input mask.")
            parser.add_argument("--input-crop", default=None, type=int, nargs=4,
                                metavar=("x", "y", "width", "height"), help="Crop input image.")
            parser.add_argument("--raw-input", action="store_true",
                                help="Skip automatic input conversion to 3-channel image.")
        except ArgumentError as ex:
            if ex.message.startswith("conflicting"):
                return
            raise ex
