from enum import Enum
from typing import Sequence, Dict, Optional, Union

from visiongraph.data.Asset import Asset
from visiongraph.estimator.BaseVisionEngine import BaseVisionEngine
from visiongraph.estimator.onnx.ONNXVisionEngine import ONNXVisionEngine
from visiongraph.estimator.openvino.VisionInferenceEngine import VisionInferenceEngine


class InferenceEngine(Enum):
    ONNX = ONNXVisionEngine
    OpenVINO = VisionInferenceEngine


class InferenceEngineFactory:
    @staticmethod
    def create(engine: InferenceEngine, assets: Sequence[Asset],
               flip_channels: bool = True,
               scale: Optional[Union[float, Sequence[float]]] = None,
               mean: Optional[Union[float, Sequence[float]]] = None,
               padding: bool = False,
               transpose: bool = True,
               **engine_options: Dict) -> BaseVisionEngine:
        if len(assets) < 0:
            raise Exception("No model or weights provided for vision engine! At least one is required!")

        instance = engine.value(*assets, flip_channels=flip_channels, scale=scale, mean=mean,
                                padding=padding, **engine_options)
        instance.transpose = transpose
        return instance
