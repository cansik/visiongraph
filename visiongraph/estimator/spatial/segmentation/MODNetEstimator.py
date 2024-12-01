from argparse import ArgumentParser, Namespace
from enum import Enum
from typing import Optional

import cv2
import numpy as np

from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.estimator.engine.InferenceEngineFactory import InferenceEngine, InferenceEngineFactory
from visiongraph.estimator.spatial.InstanceSegmentationEstimator import InstanceSegmentationEstimator
from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.InstanceSegmentationResult import InstanceSegmentationResult


class ModNetConfig(Enum):
    ModNetBasic = RepositoryAsset("modnet.onnx")
    """
    Enum representing the configuration for ModNet.
    Contains the path to the basic ModNet model asset.
    """


class ModNetEstimator(InstanceSegmentationEstimator[InstanceSegmentationResult]):
    """
    ModNetEstimator is a specific implementation of InstanceSegmentationEstimator
    tailored for the ModNet architecture.

    :param assets: Assets to be used for inference.
    :param engine: The inference engine to be used. Default is ONNX.
    """

    def __init__(self, *assets: Asset,
                 engine: InferenceEngine = InferenceEngine.ONNX):
        super().__init__(0.5)

        if len(assets) == 0:
            assets = [ModNetConfig.ModNetBasic]

        self.engine = InferenceEngineFactory.create(engine, assets,
                                                    flip_channels=True,
                                                    mean=127.0,
                                                    scale=127.0,
                                                    transpose=True,
                                                    padding=False)

        self.reference_size: int = 512
        self.mask_threshold: Optional[int] = 127

    def setup(self):
        """
        Prepares the inference engine for processing.
        """
        self.engine.setup()

    def process(self, image: np.ndarray) -> ResultList[InstanceSegmentationResult]:
        """
        Processes an input image to perform instance segmentation.

        :param image: The input image for segmentation.

        :return: A list of segmentation results containing masks and bounding boxes.
        """
        h, w = image.shape[:2]
        im_rw, im_rh = self._get_scale_factor(h, w, self.reference_size)

        self.engine.set_dynamic_input_shape("input", 1, 3, im_rh, im_rw)

        outputs = self.engine.process(image)
        output = outputs[self.engine.output_names[0]].squeeze()

        mask = cv2.resize(output, (w, h))
        mask = (mask * 255).astype(np.uint8)

        if self.mask_threshold:
            ret, mask = cv2.threshold(mask, self.mask_threshold, 255, cv2.THRESH_BINARY)

        box = BoundingBox2D(0, 0, 1, 1)
        return ResultList([InstanceSegmentationResult(0, "human", 1.0, mask, box)])

    def release(self):
        """
        Releases resources held by the inference engine.
        """
        self.engine.release()

    @staticmethod
    def _get_scale_factor(im_h, im_w, ref_size):
        """
        Computes the scaling factors for the input image dimensions.

        :param im_h: Height of the input image.
        :param im_w: Width of the input image.
        :param ref_size: Reference size for scaling.

        :return: Scaled width and height.
        """
        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            elif im_w < im_h:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w

        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32

        return im_rw, im_rh

    def configure(self, args: Namespace):
        """
        Configures the estimator with command line arguments.

        :param args: The parsed command line arguments.
        """
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        """
        Adds command line parameters for the estimator to the argument parser.

        :param parser: The argument parser to which parameters should be added.
        """
        pass

    @staticmethod
    def create(config: ModNetConfig = ModNetConfig.ModNetBasic) -> "ModNetEstimator":
        """
        Creates an instance of ModNetEstimator with the specified configuration.

        :param config: The configuration for the ModNet estimator.

        :return: An instance of the ModNetEstimator.
        """
        model = config.value
        return ModNetEstimator(model)
