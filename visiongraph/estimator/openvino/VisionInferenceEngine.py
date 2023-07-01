import logging
from typing import Dict, Optional, Any, Sequence, Union

import numpy as np
from openvino.inference_engine import IECore, IENetwork, ExecutableNetwork

from visiongraph.data.Asset import Asset
from visiongraph.estimator.BaseVisionEngine import BaseVisionEngine
from visiongraph.model.VisionEngineOutput import VisionEngineOutput


class VisionInferenceEngine(BaseVisionEngine):
    def __init__(self, model: Asset, weights: Optional[Asset] = None, flip_channels: bool = True,
                 scale: Optional[Union[float, Sequence[float]]] = None,
                 mean: Optional[Union[float, Sequence[float]]] = None,
                 padding: bool = False, device: str = "AUTO"):
        super().__init__(flip_channels, scale, mean, padding)

        self.device = device

        self.model = model
        self.weights = weights

        self.ie: Optional[IECore] = None
        self.net: Optional[IENetwork] = None
        self.infer_network: Optional[ExecutableNetwork] = None

    def setup(self):
        # setup inference engine
        self.ie = IECore()

        if self.weights is None:
            self.net = self.ie.read_network(model=self.model.path)
        else:
            self.net = self.ie.read_network(model=self.model.path, weights=self.weights.path)

        self.input_names = list(self.net.input_info.keys())
        self.output_names = list(self.net.outputs.keys())

        try:
            self.infer_network = self.ie.load_network(network=self.net, device_name=self.device)
        except RuntimeError as ex:
            logging.warning(f"Could not load network: {ex}")

            if self.device != "CPU":
                logging.warning(f"Trying to load network with CPU device directly")
                self.infer_network = self.ie.load_network(network=self.net, device_name="CPU")

    def _inference(self, image: np.ndarray, inputs: Optional[Dict[str, Any]] = None) -> VisionEngineOutput:
        return VisionEngineOutput(self.infer_network.infer(inputs=inputs))

    def get_input_shape(self, input_name: str) -> Sequence[int]:
        if input_name in self.dynamic_input_shapes:
            return self.dynamic_input_shapes[input_name]

        return self.net.input_info[input_name].input_data.shape

    def release(self):
        pass

    def get_device_name(self) -> str:
        device_name = self.ie.get_config(self.device, "FULL_DEVICE_NAME")
        return f"{device_name}"
