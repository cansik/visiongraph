from typing import Optional, List

import numpy as np
from openvino.inference_engine import IECore, IENetwork, ExecutableNetwork

from visiongraph.external.intel.model import Model


class SyncInferencePipeline:
    def __init__(self, model: Model, device: str = "CPU", ie: Optional[IECore] = None):
        self.device = device
        self.model = model

        self.ie: Optional[IECore] = ie
        self.net: Optional[IENetwork] = None
        self.infer_network: Optional[ExecutableNetwork] = None
        self.input_name: List[str] = []
        self.output_names: List[str] = []

    def setup(self):
        # setup inference engine
        if self.ie is None:
            self.ie = IECore()
        self.net = self.model.net
        self.infer_network = self.ie.load_network(network=self.net, device_name=self.device)

    def process(self, data: np.ndarray) -> List:
        inputs, preprocessing_meta = self.model.preprocess(data)
        raw_result = self.infer_network.infer(inputs=inputs)
        outputs = self.model.postprocess(raw_result, preprocessing_meta)
        return outputs

    def release(self):
        pass
