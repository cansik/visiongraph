from typing import Dict, Optional, Any, Sequence, Union, List

import openvino as ov

from visiongraph.data.Asset import Asset
from visiongraph.estimator.BaseVisionEngine import BaseVisionEngine
from visiongraph.model.VisionEngineModelLayer import VisionEngineModelLayer
from visiongraph.model.VisionEngineOutput import VisionEngineOutput


class OpenVinoEngine(BaseVisionEngine):
    """
    A class representing an OpenVINO engine (API 2.0) with inference capabilities.
    """

    def __init__(self, model: Asset,
                 weights: Optional[Asset] = None,
                 flip_channels: bool = True,
                 scale: Optional[Union[float, Sequence[float]]] = None,
                 mean: Optional[Union[float, Sequence[float]]] = None,
                 padding: bool = False, device: str = "AUTO", **config):
        """
        Initializes the OpenVINO engine.

        :param model: The model to be used for inference.
        :param weights: The weights file. Defaults to None.
        :param flip_channels: Whether to flip channels. Defaults to True.
        :param scale: Scaling factor(s). Defaults to None.
        :param mean: Mean value(s). Defaults to None.
        :param padding: Padding configuration. Defaults to False.
        :param device: Device name. Defaults to "AUTO".
        """
        super().__init__(flip_channels, scale, mean, padding)

        self.device = device

        self.model: Asset = model
        self.weights: Optional[Asset] = weights
        self.config: Dict = config

        self.ie: Optional[ov.Core] = None
        self.parsed_model: Optional[ov.Model] = None
        self.compiled_model: Optional[ov.CompiledModel] = None

        self._input_lut = {}

    def setup(self):
        """
        Sets up the inference engine.
        """
        # setup inference engine
        if self.ie is None:
            self.ie = ov.Core()

        if self.weights is None:
            self.parsed_model = self.ie.read_model(self.model.path)
        else:
            self.parsed_model = self.ie.read_model(self.model.path, self.weights.path)

        self.compiled_model = self.ie.compile_model(self.parsed_model, device_name=self.device, config=self.config)

        self.input_names = list([layer.any_name for layer in self.compiled_model.inputs])
        self.output_names = list([layer.any_name for layer in self.compiled_model.outputs])

        self._input_lut = {l.any_name: l for l in self.compiled_model.inputs}

    def predict(self, inputs: Optional[Dict[str, Any]] = None) -> VisionEngineOutput:
        """
        Performs inference on the input image.

        :param inputs: Input data. Defaults to None.

        :return: The output of the model.
        """
        request: ov.InferRequest = self.compiled_model.create_infer_request()
        request.infer(inputs=inputs)
        outputs = {l: request.output_tensors[i].data for i, l in enumerate(self.output_names)}
        return VisionEngineOutput(outputs)

    def get_input_shape(self, input_name: str) -> Sequence[int]:
        """
        Gets the shape of an input layer.

        :param input_name: The name of the input layer.

        :return: The shape of the input layer.
        """
        if input_name in self.dynamic_input_shapes:
            return self.dynamic_input_shapes[input_name]

        return tuple(self._input_lut[input_name].shape)

    def release(self):
        pass

    def get_device_name(self) -> str:
        """
        Gets the device name.

        :return: The device name.
        """
        device_name = self.ie.get_property(self.device, "FULL_DEVICE_NAME")
        return f"{device_name}"

    def get_input_layers(self) -> List[VisionEngineModelLayer]:
        """
        Gets the input layers of the model.

        :return: The input layers.
        """
        return self._get_model_layer(self.compiled_model.inputs)

    def get_output_layers(self) -> List[VisionEngineModelLayer]:
        """
        Gets the output layers of the model.

        :return: The output layers.
        """
        return self._get_model_layer(self.compiled_model.outputs)

    @staticmethod
    def _get_model_layer(compiled_layers: List) -> List[VisionEngineModelLayer]:
        """
        Gets a list of VisionEngineModelLayer objects from compiled layers.

        :param compiled_layers: The compiled layers.

        :return: The VisionEngineModelLayer objects.
        """
        return [
            VisionEngineModelLayer(name=l.any_name,
                                   shape=list(l.partial_shape),
                                   numpy_dtype=l.element_type.to_dtype(),
                                   layer_names=list(l.names))
            for l in compiled_layers
        ]
