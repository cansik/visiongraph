"""
 Copyright (c) 2021-2023 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import abc
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


@dataclass
class Metadata:
    names: Set[str] = field(default_factory=set)
    shape: List[int] = field(default_factory=list)
    layout: str = ""
    precision: str = ""
    type: str = ""
    meta: Dict = field(default_factory=dict)


class InferenceAdapter(metaclass=abc.ABCMeta):
    """
    An abstract Model Adapter with the following interface:

        - Reading the model from disk or other place
        - Loading the model to the device
        - Accessing the information about inputs/outputs
        - The model reshaping
        - Synchronous model inference
        - Asynchronous model inference
    """

    precisions = ("FP32", "I32", "FP16", "I16", "I8", "U8")

    @abc.abstractmethod
    def __init__(self):
        """
        An abstract Model Adapter constructor.
        Reads the model from disk or other place.
        """

    @abc.abstractmethod
    def load_model(self):
        """
        Loads the model on the device.
        """

    @abc.abstractmethod
    def get_input_layers(self):
        """
        Gets the names of model inputs and for each one creates the Metadata structure,
           which contains the information about the input shape, layout, precision
           in OpenVINO format, meta (optional)

        """

    @abc.abstractmethod
    def get_output_layers(self):
        """
        Gets the names of model outputs and for each one creates the Metadata structure,
           which contains the information about the output shape, layout, precision
           in OpenVINO format, meta (optional)

        """

    @abc.abstractmethod
    def reshape_model(self, new_shape):
        """
        Reshapes the model inputs to fit the new input shape.

        """

    @abc.abstractmethod
    def infer_sync(self, dict_data):
        """
        Performs the synchronous model inference. The infer is a blocking method.


        """

    @abc.abstractmethod
    def infer_async(self, dict_data, callback_fn, callback_data):
        """
        Performs the asynchronous model inference and sets
        the callback for inference completion. Also, it should
        define get_raw_result() function, which handles the result
        of inference from the model.

        """

    @abc.abstractmethod
    def is_ready(self):
        """
        In case of asynchronous execution checks if one can submit input data
        to the model for inference, or all infer requests are busy.

        """

    @abc.abstractmethod
    def await_all(self):
        """
        In case of asynchronous execution waits the completion of all
        busy infer requests.
        """

    @abc.abstractmethod
    def await_any(self):
        """
        In case of asynchronous execution waits the completion of any
        busy infer request until it becomes available for the data submission.
        """

    @abc.abstractmethod
    def get_rt_info(self, path):
        """
        Forwards to openvino.runtime.Model.get_rt_info(path)
        """

    @abc.abstractmethod
    def embed_preprocessing(
        self,
        layout=None,
        resize_mode: str = None,
        interpolation_mode="LINEAR",
        target_shape: Tuple[int] = None,
        dtype=type(int),
        brg2rgb=False,
        input_idx=0,
    ):
        """
        Embeds preprocessing into the model using OpenVINO preprocessing API
        """
