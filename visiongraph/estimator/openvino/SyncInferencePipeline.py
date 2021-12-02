from argparse import ArgumentParser, Namespace
from typing import Dict, Optional, List

import cv2
import numpy as np
from openvino.inference_engine import IECore, IENetwork, ExecutableNetwork

from visiongraph.estimator.VisionEstimator import VisionEstimator
from visiongraph.external.intel.model import Model


class SyncInferencePipeline(VisionEstimator):
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
        self.ie = IECore()
        self.net = self.model.net
        self.infer_network = self.ie.load_network(network=self.net, device_name=self.device)

    def estimate(self, image: np.ndarray, **kwargs) -> List:
        inputs, preprocessing_meta = self.model.preprocess(image)
        raw_result = self.infer_network.infer(inputs=inputs)
        outputs = self.model.postprocess(raw_result, preprocessing_meta)
        return outputs

    def release(self):
        pass

    def configure(self, args: Namespace):
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass
