from enum import Enum
from functools import partial
from typing import Sequence, Optional, Union, Any

from visiongraph.data.Asset import Asset
from visiongraph.estimator.BaseVisionEngine import BaseVisionEngine
from visiongraph.model.types.InputShapeOrder import InputShapeOrder


def _get_onnx_vision_engine_type():
    """
    Returns the ONNXVisionEngine class type.

    Returns:
        type: The ONNXVisionEngine class type.
    """
    from visiongraph.estimator.onnx.ONNXVisionEngine import ONNXVisionEngine
    return ONNXVisionEngine


def _get_open_vino_engine_type():
    """
    Returns the OpenVinoEngine class type.

    Returns:
        type: The OpenVinoEngine class type.
    """
    from visiongraph.estimator.openvino.OpenVinoEngine import OpenVinoEngine
    return OpenVinoEngine


class InferenceEngine(Enum):
    """
    Enum for selecting different inference engine types.

    Available options are:
    - ONNX: ONNXVisionEngine
    - OpenVINO: OpenVinoEngine
    - OpenVINO2: OpenVinoEngine (deprecated, use OpenVINO)
    """
    ONNX = partial(_get_onnx_vision_engine_type)
    OpenVINO = partial(_get_open_vino_engine_type)
    OpenVINO2 = partial(_get_open_vino_engine_type)


class InferenceEngineFactory:
    @staticmethod
    def create(
            engine: InferenceEngine,
            assets: Sequence[Asset],
            flip_channels: bool = True,
            scale: Optional[Union[float, Sequence[float]]] = None,
            mean: Optional[Union[float, Sequence[float]]] = None,
            padding: bool = False,
            transpose: bool = True,
            order: InputShapeOrder = InputShapeOrder.NCHW,
            **engine_options: Any
    ) -> BaseVisionEngine:
        """
        Creates an instance of a vision engine based on the selected inference engine type.

        Args:
            engine (Enum): The inference engine type to use. Can be ONNX, OpenVINO, or OpenVINO2.
            assets (Sequence[Asset]): A list of assets (e.g., models and weights) required for the chosen engine.
            flip_channels (bool, optional): Whether to flip channels in the input data. Defaults to True.
            scale (Optional[Union[float, Sequence[float]]], optional): Scaling factors for the input data. Defaults to None.
            mean (Optional[Union[float, Sequence[float]]], optional): Mean values for the input data. Defaults to None.
            padding (bool, optional): Whether to pad the input data. Defaults to False.
            transpose (bool, optional): Whether to transpose the output of the engine. Defaults to True.
            order (InputShapeOrder, optional): The input shape order to use. Defaults to NCHW.
            **engine_options: Any additional options to pass to the engine instance.

        Returns:
            BaseVisionEngine: An instance of the chosen vision engine.

        Raises:
            Exception: If no assets are provided for the selected engine.
        """
        if len(assets) < 0:
            raise Exception("No model or weights provided for vision engine! At least one is required!")

        engine_type = engine.value()
        instance = engine_type(*assets, flip_channels=flip_channels, scale=scale, mean=mean,
                               padding=padding, **engine_options)
        instance.transpose = transpose
        instance.order = order
        return instance
