from argparse import ArgumentParser, Namespace
from enum import Enum

import cv2
import numpy as np

from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.estimator.inpaint.BaseInpainter import BaseInpainter
from visiongraph.estimator.openvino.VisionInferenceEngine import VisionInferenceEngine
from visiongraph.result.ImageResult import ImageResult


class GMCNNConfig(Enum):
    GMCNN_Places2_FP16 = RepositoryAsset.openVino("gmcnn-places2-tf-fp16")
    GMCNN_Places2_FP32 = RepositoryAsset.openVino("gmcnn-places2-tf-fp32")


class GMCNNInpainter(BaseInpainter):
    def __init__(self, model: Asset, weights: Asset, device: str = "CPU"):
        super().__init__()
        self.engine = VisionInferenceEngine(model, weights, device=device)

    def setup(self):
        self.engine.setup()

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> ImageResult:
        # prepare mask
        _, binary_mask = cv2.threshold(mask, 1, 1, cv2.THRESH_BINARY)

        mask_input_name = self.engine.input_names[1]
        mask_input = self.engine.pre_process_image(binary_mask, mask_input_name, flip_channels=False)
        inputs = {mask_input_name: mask_input}

        outputs = self.engine.process(image, inputs)
        output = outputs[self.engine.output_names[0]]

        reconstructed_frame = np.transpose(output, (0, 2, 3, 1)).astype(np.uint8)
        reconstructed_frame = reconstructed_frame.reshape(reconstructed_frame.shape[1:])

        resized_frame = cv2.resize(reconstructed_frame, (image.shape[1], image.shape[0]))

        return ImageResult(resized_frame)

    def release(self):
        self.engine.release()

    def configure(self, args: Namespace):
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass

    @staticmethod
    def create(config: GMCNNConfig = GMCNNConfig.GMCNN_Places2_FP32) -> "GMCNNInpainter":
        model, weights = config.value
        return GMCNNInpainter(model, weights)
