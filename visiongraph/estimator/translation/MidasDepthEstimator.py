from argparse import ArgumentParser, Namespace
from enum import Enum
from typing import Optional

import cv2
import numpy as np
import onnxruntime as rt

from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.estimator.translation.DepthEstimator import DepthEstimator
from visiongraph.external.midas.transforms import Resize, PrepareForNet
from visiongraph.result.DepthMap import DepthMap


class MidasConfig(Enum):
    MidasSmall = (RepositoryAsset("model-small.simp.onnx"), 256)


class MidasDepthEstimator(DepthEstimator):
    def __init__(self, model: Asset, net_size: int = 256):
        super().__init__()

        self.model_asset = model
        self.net_size = net_size

        self.model: Optional[rt.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.output_name: Optional[str] = None

        resize_image = Resize(
            self.net_size,
            self.net_size,
            resize_target=None,
            keep_aspect_ratio=False,
            ensure_multiple_of=32,
            resize_method="upper_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        )

        self.transform = self.compose2(resize_image, PrepareForNet())

    def setup(self):
        self.model = rt.InferenceSession(self.model_asset.path, providers=["CUDAExecutionProvider",
                                                                           "OpenVINOExecutionProvider",
                                                                           "CPUExecutionProvider"])
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name

    def process(self, image: np.ndarray) -> DepthMap:
        # todo: maybe convert COLOR_BGR2RGB
        normalized_image = image / 255.0
        img_input = self.transform({"image": normalized_image})["image"]

        # compute
        output = self.model.run([self.output_name],
                                {self.input_name: img_input.reshape(1, 3, self.net_size, self.net_size).astype(
                                    np.float32)})[0]

        prediction = np.array(output).reshape(self.net_size, self.net_size)
        depth = cv2.resize(prediction, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)

        return DepthMap(depth)

    def release(self):
        pass

    def configure(self, args: Namespace):
        super().configure(args)

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass

    @staticmethod
    def compose2(f1, f2):
        return lambda x: f2(f1(x))

    @staticmethod
    def create(config: MidasConfig = MidasConfig.MidasSmall) -> "MidasDepthEstimator":
        model, size = config.value
        return MidasDepthEstimator(model, size)
