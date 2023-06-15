import logging
from enum import Enum
from functools import partial
from typing import Sequence, Dict, Optional, Union, Any

from visiongraph.data.Asset import Asset
from visiongraph.estimator.BaseVisionEngine import BaseVisionEngine
from visiongraph.model.types.InputShapeOrder import InputShapeOrder


def _get_onnx_vision_engine_type():
    from visiongraph.estimator.onnx.ONNXVisionEngine import ONNXVisionEngine
    return ONNXVisionEngine


def _get_vision_inference_engine_type():
    from visiongraph.estimator.openvino.VisionInferenceEngine import VisionInferenceEngine
    return VisionInferenceEngine


def _get_open_vino_engine_type():
    from visiongraph.estimator.openvino.OpenVinoEngine import OpenVinoEngine
    return OpenVinoEngine


class InferenceEngine(Enum):
    ONNX = partial(_get_onnx_vision_engine_type)
    OpenVINO = partial(_get_vision_inference_engine_type)
    OpenVINO2 = partial(_get_open_vino_engine_type)


class InferenceEngineFactory:
    @staticmethod
    def create(engine: InferenceEngine, assets: Sequence[Asset],
               flip_channels: bool = True,
               scale: Optional[Union[float, Sequence[float]]] = None,
               mean: Optional[Union[float, Sequence[float]]] = None,
               padding: bool = False,
               transpose: bool = True,
               order: InputShapeOrder = InputShapeOrder.NCHW,
               **engine_options: Any) -> BaseVisionEngine:
        if len(assets) < 0:
            raise Exception("No model or weights provided for vision engine! At least one is required!")

        engine_type = engine.value()
        instance = engine_type(*assets, flip_channels=flip_channels, scale=scale, mean=mean,
                               padding=padding, **engine_options)
        instance.transpose = transpose
        instance.order = order
        return instance
