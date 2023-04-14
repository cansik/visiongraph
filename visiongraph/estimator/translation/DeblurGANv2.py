from argparse import ArgumentParser, Namespace
from enum import Enum

import cv2
import numpy as np

from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.estimator.VisionEstimator import VisionEstimator
from visiongraph.estimator.openvino.OpenVinoEngine import OpenVinoEngine
from visiongraph.estimator.openvino.VisionInferenceEngine import VisionInferenceEngine
from visiongraph.result.ImageResult import ImageResult


class DeblurGANv2Config(Enum):
    DeblurGANv2_FP16 = RepositoryAsset.openVino("deblurgan-v2-fp16")
    DeblurGANv2_FP32 = RepositoryAsset.openVino("deblurgan-v2-fp32")


class DeblurGANv2(VisionEstimator[ImageResult]):
    def __init__(self, model: Asset, weights: Asset):
        self.engine = OpenVinoEngine(model, weights)

    def setup(self):
        self.engine.setup()

    def process(self, data: np.ndarray) -> ImageResult:
        outputs = self.engine.process(data)
        output = outputs[self.engine.output_names[0]]

        reconstructed_frame = (np.transpose(output, (0, 2, 3, 1)) * 255.0).astype(np.uint8)
        reconstructed_frame = reconstructed_frame.reshape(reconstructed_frame.shape[1:])
        resized_frame = cv2.resize(reconstructed_frame, (data.shape[1], data.shape[0]))

        return ImageResult(resized_frame)

    def release(self):
        self.engine.release()

    def configure(self, args: Namespace):
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass

    @staticmethod
    def create(config: DeblurGANv2Config = DeblurGANv2Config.DeblurGANv2_FP32) -> "DeblurGANv2":
        model, weights = config.value
        return DeblurGANv2(model, weights)
