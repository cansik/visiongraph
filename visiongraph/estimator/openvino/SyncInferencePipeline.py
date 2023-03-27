from typing import Optional, List

import numpy as np
from openvino.inference_engine import IECore, IENetwork, ExecutableNetwork

from visiongraph.external.intel.models.model import Model


class SyncInferencePipeline:
    def __init__(self, model: Model, device: str = "AUTO"):
        self.device = device
        self.model = model

    def setup(self):
        if not self.model.model_loaded:
            self.model.load()

    def process(self, data: np.ndarray) -> List:
        inputs, preprocessing_meta = self.model.preprocess(data)
        raw_result = self.model.infer_sync(inputs)
        outputs = self.model.postprocess(raw_result, preprocessing_meta)
        return outputs

    def release(self):
        pass
