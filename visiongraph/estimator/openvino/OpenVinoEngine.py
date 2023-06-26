from typing import Dict, Optional, Any, Sequence, Union

import numpy as np
import openvino.runtime as ov

from visiongraph.data.Asset import Asset
from visiongraph.estimator.BaseVisionEngine import BaseVisionEngine
from visiongraph.model.VisionEngineOutput import VisionEngineOutput


class OpenVinoEngine(BaseVisionEngine):

    def __init__(self, model: Asset,
                 weights: Optional[Asset] = None,
                 flip_channels: bool = True,
                 scale: Optional[Union[float, Sequence[float]]] = None,
                 mean: Optional[Union[float, Sequence[float]]] = None,
                 padding: bool = False, device: str = "AUTO", **config):
        super().__init__(flip_channels, scale, mean, padding)

        self.device = device

        self.model: Asset = model
        self.weights: Optional[Asset] = weights
        self.config: Dict = config

        self.ie: Optional[ov.Core] = None
        self.parsed_model: Optional = None
        self.compiled_model: Optional[ov.CompiledModel] = None

        self._input_lut = {}

    def setup(self):
        # setup inference engine
        if self.ie is None:
            self.ie = ov.Core()

        if self.weights is None:
            self.parsed_model = self.ie.read_model(self.model.path)
        else:
            self.parsed_model = self.ie.read_model(self.model.path, self.weights.path)

        self.compiled_model = self.ie.compile_model(self.parsed_model, device_name=self.device, config=self.config)

        self.input_names = list([l.any_name for l in self.compiled_model.inputs])
        self.output_names = list([l.any_name for l in self.compiled_model.outputs])

        self._input_lut = {l.any_name: l for l in self.compiled_model.inputs}

    def _inference(self, image: np.ndarray, inputs: Optional[Dict[str, Any]] = None) -> VisionEngineOutput:
        request: ov.InferRequest = self.compiled_model.create_infer_request()
        request.infer(inputs=inputs)
        outputs = {l: request.outputs[i].data for i, l in enumerate(self.output_names)}
        return VisionEngineOutput(outputs)

    def get_input_shape(self, input_name: str) -> Sequence[int]:
        if input_name in self.dynamic_input_shapes:
            return self.dynamic_input_shapes[input_name]

        return tuple(self._input_lut[input_name].shape)

    def release(self):
        pass

    def get_device_name(self) -> str:
        device_name = self.ie.get_property(self.device, "FULL_DEVICE_NAME")
        return f"{device_name}"
