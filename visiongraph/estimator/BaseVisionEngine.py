from abc import abstractmethod, ABC
from typing import Dict, Optional, List, Any, Sequence, Tuple, Union

import cv2
import numpy as np

from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D

PADDING_BOX_OUTPUT_NAME = "padding-box"


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

    def process(self, image: np.ndarray, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
        in_frame, bbox = self.pre_process_image(image, self.first_input_name,
                                                self.flip_channels, self.scale, self.mean,
                                                self.padding, self.transpose)

        if inputs is None:
            inputs = {}

        inputs.update({self.first_input_name: in_frame})
        outputs = self._inference(image, inputs)

        # add padding box
        outputs[PADDING_BOX_OUTPUT_NAME] = bbox

        return outputs

    @abstractmethod
    def _inference(self, image: np.ndarray, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
        pass

    def pre_process_image(self, image: np.ndarray, input_name: str, flip_channels: bool = True,
                          scale: Optional[Union[float, Sequence[float]]] = None,
                          mean: Optional[Union[float, Sequence[float]]] = None,
                          padding: bool = False,
                          transpose: bool = True) -> Tuple[np.ndarray, BoundingBox2D]:
        input_channels = image.shape[-1] if image.ndim == 3 else 1
        batch_size, channels, height, width = self.get_input_shape(input_name)

        if padding:
            pc = self.padding_color if self.padding_color is not None else (0, 0, 0)
            in_frame, bbox = self._resize_and_pad(image, (width, height), pc)
        else:
            in_frame = cv2.resize(image, (width, height))
            bbox = BoundingBox2D(0, 0, width, height)

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

        return in_frame, bbox.scale(1.0 / width, 1.0 / height)

    @staticmethod
    def _resize_and_pad(image: np.ndarray, new_size: Tuple[int, int],
                        color: Tuple[int, int, int] = (125, 125, 125)) -> Tuple[np.ndarray, BoundingBox2D]:
        in_h, in_w = image.shape[:2]
        new_w, new_h = new_size
        scale = min(new_w / in_w, new_h / in_h)
        scale_new_w, scale_new_h = int(in_w * scale), int(in_h * scale)
        resized_img = cv2.resize(image, (scale_new_w, scale_new_h))
        d_w = max(new_w - scale_new_w, 0)
        d_h = max(new_h - scale_new_h, 0)
        top, bottom = d_h // 2, d_h - (d_h // 2)
        left, right = d_w // 2, d_w - (d_w // 2)
        result = cv2.copyMakeBorder(resized_img, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=color)
        return result, BoundingBox2D(left, top, new_w, new_h)

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
