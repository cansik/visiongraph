from typing import Dict, Optional, List, Any, Sequence, Tuple

import cv2
import numpy as np
from openvino.inference_engine import IECore, IENetwork, ExecutableNetwork

from visiongraph.data.Asset import Asset
from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D

PADDING_BOX_OUTPUT_NAME = "padding-box"

class VisionInferenceEngine:
    def __init__(self, model: Asset, weights: Asset,
                 flip_channels: bool = True, normalize: bool = False, padding: bool = False,
                 device: str = "CPU"):

        # default params for first input
        self.flip_channels = flip_channels
        self.normalize = normalize
        self.padding = padding
        self.padding_color: Optional[Sequence[int]] = None

        self.device = device

        self.model = model
        self.weights = weights

        self.ie: Optional[IECore] = None
        self.net: Optional[IENetwork] = None
        self.input_names: List[str] = []
        self.output_names: List[str] = []
        self.infer_network: Optional[ExecutableNetwork] = None

    def setup(self):
        # setup inference engine
        self.ie = IECore()
        self.net = self.ie.read_network(model=self.model.path, weights=self.weights.path)

        self.input_names = list(self.net.input_info.keys())
        self.output_names = list(self.net.outputs.keys())

        self.infer_network = self.ie.load_network(network=self.net, device_name=self.device)

    def process(self, image: np.ndarray, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
        in_frame, bbox = self.pre_process_image(image, self.input_names[0],
                                                self.flip_channels, self.normalize, self.padding)

        if inputs is None:
            inputs = {}

        inputs.update({self.first_input_name: in_frame})
        outputs = self.infer_network.infer(inputs=inputs)

        # add padding box
        outputs[PADDING_BOX_OUTPUT_NAME] = bbox

        return outputs

    def pre_process_image(self, image: np.ndarray, input_name: str,
                          flip_channels: bool = True, normalize: bool = False,
                          padding: bool = False) -> Tuple[np.ndarray, BoundingBox2D]:
        input_channels = image.shape[-1] if image.ndim == 3 else 1
        batch_size, channels, height, width = self.get_input_shape(input_name)

        if padding:
            in_frame, bbox = self._resize_and_pad(image, width, height)
        else:
            in_frame = cv2.resize(image, (width, height))
            bbox = BoundingBox2D(0, 0, width, height)

        if input_channels == 3 and channels == 1:
            in_frame = cv2.cvtColor(in_frame, cv2.COLOR_RGB2GRAY)
        elif input_channels == 1 and channels == 3:
            in_frame = cv2.cvtColor(in_frame, cv2.COLOR_GRAY2RGB)

        if input_channels == 3 and flip_channels:
            in_frame = in_frame.transpose((2, 0, 1))

        in_frame = in_frame.reshape((1, channels, height, width))

        if normalize:
            in_frame = in_frame.astype(np.float32) / 255.0

        return in_frame, bbox.scale(1.0 / width, 1.0 / height)

    def _resize_and_pad(self, image: np.ndarray, width: int, height: int) -> Tuple[np.ndarray, BoundingBox2D]:
        # resize input image
        h, w = image.shape[:2]

        if w > h:
            nh = int(width / w * h)
            image = cv2.resize(image, (width, nh))
        else:
            nw = int(height / h * w)
            image = cv2.resize(image, (nw, height))

        h, w = image.shape[:2]

        # get final image size
        size = h if h > w else w

        # todo: check if width and height are different (non-square padding)

        #  create base image with background color
        background = np.zeros([size, size, 3], dtype=np.uint8)

        if self.padding_color is not None:
            background[:] = self.padding_color

        # add image into center
        xs = round((size - w) * 0.5)
        ys = round((size - h) * 0.5)

        background[ys:ys + h, xs:xs + w] = image

        return background, BoundingBox2D(xs, ys, w, h)

    @property
    def first_input_name(self) -> str:
        return self.input_names[0]

    def get_input_shape(self, input_name: str) -> Sequence[int]:
        return self.net.input_info[input_name].input_data.shape

    @property
    def first_input_shape(self) -> Sequence[int]:
        return self.get_input_shape(self.first_input_name)

    def release(self):
        pass
