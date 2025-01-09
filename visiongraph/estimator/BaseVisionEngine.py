from abc import abstractmethod, ABC
from typing import Dict, Optional, List, Any, Sequence, Tuple, Union

import cv2
import numpy as np

from visiongraph.model.VisionEngineModelLayer import VisionEngineModelLayer
from visiongraph.model.VisionEngineOutput import VisionEngineOutput
from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.model.geometry.Size2D import Size2D
from visiongraph.model.types.InputShapeOrder import InputShapeOrder
from visiongraph.util import ImageUtils


class BaseVisionEngine(ABC):
    """
    An abstract base class for a vision engine that processes images and performs inference.
    """

    def __init__(self, flip_channels: bool = True,
                 scale: Optional[Union[float, Sequence[float]]] = None,
                 mean: Optional[Union[float, Sequence[float]]] = None,
                 padding: bool = False,
                 transpose: bool = True,
                 order: InputShapeOrder = InputShapeOrder.NCHW,
                 dtype: np.dtype = np.float32):
        """
        Initializes the BaseVisionEngine with specified parameters.

        :param flip_channels: Whether to flip the image channels.
        :param scale: Scale factor(s) for preprocessing.
        :param mean: Mean value(s) to subtract during preprocessing.
        :param padding: Whether to apply padding to the image.
        :param transpose: Whether to transpose the image dimensions.
        :param order: The order of input shapes (e.g., NCHW or NWHC).
        :param dtype: The data type of the input images.
        """
        self.flip_channels = flip_channels
        self.scale = scale
        self.mean = mean
        self.padding = padding
        self.padding_color: Optional[Sequence[int]] = None
        self.transpose = transpose
        self.order = order
        self.dtype = dtype

        self.input_names: List[str] = []
        self.output_names: List[str] = []

        self.dynamic_input_shapes: Dict[str, List[int]] = dict()

    @abstractmethod
    def setup(self):
        """
        Sets up the vision engine. Must be implemented by subclasses.
        """
        pass

    def process(self, image: np.ndarray, inputs: Optional[Dict[str, Any]] = None) -> VisionEngineOutput:
        """
        Processes an input image and performs inference.

        :param image: The input image to be processed.
        :param inputs: Optional additional inputs for inference.

        :return: The output from the inference process.
        """
        in_frame, padding_box, image_size = self.pre_process_image(image, self.first_input_name,
                                                                   self.flip_channels, self.scale, self.mean,
                                                                   self.padding, self.transpose, self.order,
                                                                   self.dtype)

        if inputs is None:
            inputs = {}

        inputs.update({self.first_input_name: in_frame})
        outputs = self.predict(inputs)

        # add padding box
        outputs.padding_box = padding_box
        outputs.image_size = image_size

        return outputs

    @abstractmethod
    def predict(self, inputs: Optional[Dict[str, Any]] = None) -> VisionEngineOutput:
        """
        Performs inference on the input image.

        :param inputs: Inputs for inference.

        :return: The output from the inference process.
        """
        pass

    def pre_process_image(self, image: np.ndarray, input_name: str, flip_channels: bool = True,
                          scale: Optional[Union[float, Sequence[float]]] = None,
                          mean: Optional[Union[float, Sequence[float]]] = None,
                          padding: bool = False,
                          transpose: bool = True,
                          order: InputShapeOrder = InputShapeOrder.NCHW,
                          dtype: np.dtype = np.float32) -> Tuple[np.ndarray, BoundingBox2D, Size2D]:
        """
        Preprocesses the input image for inference.

        :param image: The input image to preprocess.
        :param input_name: The name of the input to be processed.
        :param flip_channels: Whether to flip the image channels.
        :param scale: Scale factor(s) for preprocessing.
        :param mean: Mean value(s) to subtract during preprocessing.
        :param padding: Whether to apply padding to the image.
        :param transpose: Whether to transpose the image dimensions.
        :param order: The order of input shapes (e.g., NCHW or NWHC).
        :param dtype: The data type of the input images.

        :return: A tuple containing the preprocessed image, the padding box, and the image size.
        """
        input_channels = image.shape[-1] if image.ndim == 3 else 1

        if order == InputShapeOrder.NWHC:
            batch_size, width, height, channels = self.get_input_shape(input_name)
        else:
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
        in_frame = in_frame.astype(dtype)

        if mean is not None:
            in_frame -= mean

        if scale is not None:
            in_frame /= scale

        # flip rgb
        if input_channels == 3 and flip_channels:
            in_frame = cv2.cvtColor(in_frame, cv2.COLOR_RGB2BGR)

        # transform to blob
        if transpose:
            if channels == 3:
                in_frame = in_frame.transpose((2, 0, 1))
            else:
                in_frame = in_frame.transpose((1, 0))

        # make nchw
        if order == InputShapeOrder.NWHC:
            in_frame = in_frame.reshape((1, width, height, channels))
        else:
            in_frame = in_frame.reshape((1, channels, height, width))

        return in_frame, pad_bbox, image_size

    def set_dynamic_input_shape(self, name: str, batch_size: int, channels: int, height: int, width: int):
        """
        Sets the dynamic input shape for a given input name.

        :param name: The name of the input.
        :param batch_size: The batch size.
        :param channels: The number of channels.
        :param height: The height of the input.
        :param width: The width of the input.
        """
        self.dynamic_input_shapes[name] = [batch_size, channels, height, width]

    @property
    def first_input_name(self) -> str:
        """
        Gets the name of the first input.

        :return: The name of the first input.
        """
        return self.input_names[0]

    @abstractmethod
    def get_input_shape(self, input_name: str) -> Sequence[int]:
        """
        Gets the shape of the specified input.

        :param input_name: The name of the input.

        :return: The shape of the input.
        """
        pass

    @abstractmethod
    def get_device_name(self) -> str:
        """
        Gets the name of the device used for processing.

        :return: The name of the processing device.
        """
        pass

    @property
    def first_input_shape(self) -> Sequence[int]:
        """
        Gets the shape of the first input.

        :return: The shape of the first input.
        """
        return self.get_input_shape(self.first_input_name)

    @abstractmethod
    def release(self):
        """
        Releases any resources held by the vision engine.
        """
        pass

    @abstractmethod
    def get_input_layers(self) -> List[VisionEngineModelLayer]:
        """
        Gets the input layers of the model.

        :return: A list of input layers.
        """
        pass

    @abstractmethod
    def get_output_layers(self) -> List[VisionEngineModelLayer]:
        """
        Gets the output layers of the model.

        :return: A list of output layers.
        """
        pass
