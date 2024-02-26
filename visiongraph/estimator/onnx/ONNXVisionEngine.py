from typing import Sequence, Optional, Dict, Any, List, Union

import numpy as np
import onnxruntime as rt

from visiongraph.data.Asset import Asset
from visiongraph.estimator.BaseVisionEngine import BaseVisionEngine
from visiongraph.model.VisionEngineModelLayer import VisionEngineModelLayer
from visiongraph.model.VisionEngineOutput import VisionEngineOutput


class ONNXVisionEngine(BaseVisionEngine):
    def __init__(self, model: Asset, execution_providers: Optional[List[str]] = None,
                 flip_channels: bool = True,
                 scale: Optional[Union[float, Sequence[float]]] = None,
                 mean: Optional[Union[float, Sequence[float]]] = None,
                 padding: bool = False):
        super().__init__(flip_channels, scale, mean, padding)

        self.model = model
        self.execution_providers = execution_providers

        self.session: Optional[rt.InferenceSession] = None
        self.session_options = rt.SessionOptions()

        self.preferred_execution_providers = ["CUDAExecutionProvider",
                                              "DmlExecutionProvider",
                                              "CPUExecutionProvider"]

        self.dtype_conversion_table = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(float32)": np.float32,
            "tensor(int64)": np.int64,
            "tensor(int32)": np.int32,
            "tensor(int8)": np.int8,
            "tensor(uint8)": np.uint8
        }

    def setup(self):
        if self.execution_providers is None:
            self.execution_providers = self.get_execution_providers()

        self.session = rt.InferenceSession(self.model.path,
                                           providers=self.execution_providers,
                                           sess_options=self.session_options)

        # read input infos
        self.input_names = [e.name for e in self.session.get_inputs()]
        self.output_names = [e.name for e in self.session.get_outputs()]

    def _inference(self, image: np.ndarray, inputs: Optional[Dict[str, Any]] = None) -> VisionEngineOutput:
        results = self.session.run(self.output_names, inputs)
        result_dict = VisionEngineOutput({n: r for n, r in zip(self.output_names, results)})
        return result_dict

    def get_input_shape(self, input_name: str) -> Sequence[int]:
        if input_name in self.dynamic_input_shapes:
            return self.dynamic_input_shapes[input_name]

        for input in self.session.get_inputs():
            if input.name == input_name:
                return input.shape

        return []

    def release(self):
        self.session = None

    def get_execution_providers(self) -> List[str]:
        providers = rt.get_available_providers()
        providers_set = set(providers)

        # filter user selected providers
        selected_providers = [p for p in self.preferred_execution_providers if p in providers_set]

        if len(selected_providers) == 0:
            return providers

        return selected_providers

    def get_device_name(self) -> str:
        return self.session.get_providers()[0]

    def get_input_layers(self) -> List[VisionEngineModelLayer]:
        return self._get_model_layer(self.session.get_inputs())

    def get_output_layers(self) -> List[VisionEngineModelLayer]:
        return self._get_model_layer(self.session.get_outputs())

    def _get_model_layer(self, compiled_layers: List[rt.NodeArg]) -> List[VisionEngineModelLayer]:
        return [
            VisionEngineModelLayer(name=l.name,
                                   shape=list(l.shape),
                                   numpy_dtype=self._to_numpy_dtype(l.type),
                                   layer_names=list(l.name))
            for l in compiled_layers
        ]

    def _to_numpy_dtype(self, type_text: str) -> np.dtype:
        if type_text not in self.dtype_conversion_table:
            raise TypeError(f"Could not convert '{type_text}' into a numpy dtype.")

        return self.dtype_conversion_table[type_text]
