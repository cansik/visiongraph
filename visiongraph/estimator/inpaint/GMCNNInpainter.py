from argparse import ArgumentParser, Namespace
from enum import Enum

import cv2
import numpy as np

from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.estimator.inpaint.BaseInpainter import BaseInpainter
from visiongraph.estimator.openvino.OpenVinoEngine import OpenVinoEngine
from visiongraph.result.ImageResult import ImageResult


class GMCNNConfig(Enum):
    """
    Enum to represent different configurations for the GMCNN model.
    """

    GMCNN_Places2_FP16 = RepositoryAsset.openVino("gmcnn-places2-tf-fp16")
    GMCNN_Places2_FP32 = RepositoryAsset.openVino("gmcnn-places2-tf-fp32")


class GMCNNInpainter(BaseInpainter):
    """
    A class representing a GMCNN-based inpainting model.

    https://github.com/shepnerd/inpainting_gmcnn
    """

    def __init__(self, model: Asset, weights: Asset, device: str = "AUTO"):
        """
        Initializes the GMCNNInpainter object.

        :param model: The repository asset containing the model.
        :param weights: The repository asset containing the weights for the model.
        :param device: The device to use. Defaults to "AUTO".
        """
        super().__init__()
        self.engine = OpenVinoEngine(model, weights, device=device, flip_channels=False)

    def setup(self):
        """
        Sets up the OpenVINO engine.
        """
        self.engine.setup()

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> ImageResult:
        """
        Performs inpainting on the given image using the provided mask.

        :param image: The input image.
        :param mask: The binary mask to use for inpainting.

        :return: The resulting image after inpainting.
        """
        # Prepare mask
        _, binary_mask = cv2.threshold(mask, 1, 1, cv2.THRESH_BINARY)

        mask_input_name = self.engine.input_names[1]
        mask_input, padding_box, image_box = self.engine.pre_process_image(binary_mask, mask_input_name,
                                                                           flip_channels=False, transpose=False)
        inputs = {mask_input_name: mask_input}

        outputs = self.engine.process(image, inputs)
        output = outputs[self.engine.output_names[0]]

        reconstructed_frame = np.transpose(output, (0, 2, 3, 1)).astype(np.uint8)
        reconstructed_frame = reconstructed_frame.reshape(reconstructed_frame.shape[1:])

        resized_frame = cv2.resize(reconstructed_frame, (image.shape[1], image.shape[0]))

        return ImageResult(resized_frame)

    def release(self):
        """
        Releases the OpenVINO engine.
        """
        self.engine.release()

    def configure(self, args: Namespace):
        """
        Configures the GMCNNInpainter object based on the provided arguments.

        :param args: The namespace containing the configuration arguments.
        """
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        """
        Adds parameters to the argument parser for configuring the GMCNNInpainter model.
        """
        pass

    @staticmethod
    def create(config: GMCNNConfig = GMCNNConfig.GMCNN_Places2_FP32) -> "GMCNNInpainter":
        """
        Creates a new instance of the GMCNNInpainter class.

        :param config: The configuration to use for the model. Defaults to GMCNNConfig.GMCNN_Places2_FP32.

        :return: A new instance of the GMCNNInpainter class.
        """
        model, weights = config.value
        return GMCNNInpainter(model, weights)
