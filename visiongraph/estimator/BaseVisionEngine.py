from abc import abstractmethod, ABC
from typing import Dict, Optional, List, Any, Sequence, Tuple, Union

import cv2
import numpy as np
import vector

from visiongraph.model.VisionEngineOutput import VisionEngineOutput
from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.model.geometry.Size2D import Size2D
from visiongraph.util import ImageUtils


class BaseVisionEngine(ABC):
    def __init__(self, flip_channels: bool = True,
                 scale: Optional[Union[float, Sequence[float]]] = None,
                 mean: Optional[Union[float, Sequence[float]]] = None,
                 padding: bool = False,
                 transpose: bool = True):

        self.flip_channels = flip_channels
        self.scale = scale
        self.mean = mean
        self.padding = padding
        self.padding_color: Optional[Sequence[int]] = None
        self.transpose = transpose

        self.input_names: List[str] = []
        self.output_names: List[str] = []

    @abstractmethod
    def setup(self):
        pass

    def process(self, image: np.ndarray, inputs: Optional[Dict[str, Any]] = None) -> VisionEngineOutput:
        in_frame, padding_box, image_size = self.pre_process_image(image, self.first_input_name,
                                                                   self.flip_channels, self.scale, self.mean,
                                                                   self.padding, self.transpose)

        if inputs is None:
            inputs = {}

        inputs.update({self.first_input_name: in_frame})
        outputs = self._inference(image, inputs)

        # add padding box
        outputs.padding_box = padding_box
        outputs.image_size = image_size

        return outputs

    @abstractmethod
    def _inference(self, image: np.ndarray, inputs: Optional[Dict[str, Any]] = None) -> VisionEngineOutput:
        pass

    def pre_process_image(self, image: np.ndarray, input_name: str, flip_channels: bool = True,
                          scale: Optional[Union[float, Sequence[float]]] = None,
                          mean: Optional[Union[float, Sequence[float]]] = None,
                          padding: bool = False,
                          transpose: bool = True) -> Tuple[np.ndarray, BoundingBox2D, Size2D]:
        input_channels = image.shape[-1] if image.ndim == 3 else 1
        batch_size, channels, height, width = self.get_input_shape(input_name)

        if padding:
            pc = self.padding_color if self.padding_color is not None else (0, 0, 0)
            in_frame, pad_bbox = ImageUtils.resize_and_pad(image, (width, height), pc)
        else:
            in_frame = cv2.resize(image, (width, height))
            pad_bbox = BoundingBox2D(0, 0, width, height)

        image_size = Size2D.from_image(in_frame)

        if input_channels == 3 and channels == 1:
            in_frame = cv2.cvtColor(in_frame, cv2.COLOR_RGB2GRAY)
        elif input_channels == 1 and channels == 3:
            in_frame = cv2.cvtColor(in_frame, cv2.COLOR_GRAY2RGB)

        # convert to float32
        in_frame = in_frame.astype(np.float32)

        if mean is not None:
            in_frame -= mean

        if scale is not None:
            in_frame /= scale

        # flip rgb
        if input_channels == 3 and flip_channels:
            in_frame = cv2.cvtColor(in_frame, cv2.COLOR_RGB2BGR)

        # transform to blob
        if transpose:
            in_frame = in_frame.transpose((2, 0, 1))

        # make ncwh
        in_frame = in_frame.reshape((1, channels, height, width))

        return in_frame, pad_bbox, image_size

    @property
    def first_input_name(self) -> str:
        return self.input_names[0]

    @abstractmethod
    def get_input_shape(self, input_name: str) -> Sequence[int]:
        pass

    @property
    def first_input_shape(self) -> Sequence[int]:
        return self.get_input_shape(self.first_input_name)

    @abstractmethod
    def release(self):
        pass
