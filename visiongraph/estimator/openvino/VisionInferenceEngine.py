from typing import Dict, Optional, List

import cv2
import numpy as np
from openvino.inference_engine import IECore, IENetwork, ExecutableNetwork

from visiongraph.data.Asset import Asset


class VisionInferenceEngine:
    def __init__(self, model: Asset, weights: Asset,
                 batch_size: Optional[int] = None, channels: Optional[int] = None,
                 width: Optional[int] = None, height: Optional[int] = None,
                 flip_channels: bool = True, normalize: bool = False,
                 device: str = "CPU"):

        self.batch_size: Optional[int] = batch_size
        self.channels: Optional[int] = channels
        self.width: Optional[int] = width
        self.height: Optional[int] = height

        self.flip_channels = flip_channels
        self.normalize = normalize
        self.device = device

        self.model = model
        self.weights = weights

        self.ie: Optional[IECore] = None
        self.net: Optional[IENetwork] = None
        self.input_name: List[str] = []
        self.output_names: List[str] = []
        self.infer_network: Optional[ExecutableNetwork] = None

    def setup(self):
        # setup inference engine
        self.ie = IECore()
        self.net = self.ie.read_network(model=self.model.path, weights=self.weights.path)

        self.input_name = list(self.net.input_info.keys())[0]

        # auto infer width and height from model
        input_info = self.net.input_info[self.input_name]
        input_shape = input_info.input_data.shape

        if self.batch_size is None:
            self.batch_size = input_shape[0]

        if self.channels is None:
            self.channels = input_shape[1]

        if self.height is None:
            self.height = input_shape[2]

        if self.width is None:
            self.width = input_shape[3]

        self.output_names = list(self.net.outputs.keys())
        self.infer_network = self.ie.load_network(network=self.net, device_name=self.device)

    def process(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        in_frame = cv2.resize(image, (self.width, self.height))

        if self.flip_channels:
            in_frame = in_frame.transpose((2, 0, 1))

        in_frame = in_frame.reshape((1, self.channels, self.height, self.width))

        if self.normalize:
            in_frame = in_frame.astype('float32') / 255.0

        outputs = self.infer_network.infer(inputs={self.input_name: in_frame})
        return outputs

    def release(self):
        pass
