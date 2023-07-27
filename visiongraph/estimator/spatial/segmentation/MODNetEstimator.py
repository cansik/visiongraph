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


class ModNetEstimator(InstanceSegmentationEstimator[InstanceSegmentationResult]):
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
        self.engine.setup()

    def process(self, image: np.ndarray) -> ResultList[InstanceSegmentationResult]:
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
        self.engine.release()

    @staticmethod
    def _get_scale_factor(im_h, im_w, ref_size):

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
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass

    @staticmethod
    def create(config: ModNetConfig = ModNetConfig.ModNetBasic) -> "ModNetEstimator":
        model = config.value
        return ModNetEstimator(model)
