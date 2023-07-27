from argparse import ArgumentParser, Namespace
from enum import Enum

import cv2
import numpy as np

from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.estimator.VisionEstimator import VisionEstimator
from visiongraph.estimator.engine.InferenceEngineFactory import InferenceEngine, InferenceEngineFactory
from visiongraph.result.ImageResult import ImageResult


class MBLLENConfig(Enum):
    MBLLEN_Syn_LowLight_Noise_720x1280 = RepositoryAsset("MBLLEN-syn-lowlight-noise-720x1280.onnx")
    MBLLEN_Syn_LowLight_Noise_480x640 = RepositoryAsset("MBLLEN-syn-lowlight-noise-480x640.onnx")
    MBLLEN_Syn_LowLight_Noise_360x640 = RepositoryAsset("MBLLEN-syn-lowlight-noise-360x640.onnx")
    MBLLEN_Syn_LowLight_Noise_240x320 = RepositoryAsset("MBLLEN-syn-lowlight-noise-240x320.onnx")
    MBLLEN_Syn_LowLight_Noise_180x320 = RepositoryAsset("MBLLEN-syn-lowlight-noise-180x320.onnx")


class MBLLENEstimator(VisionEstimator[ImageResult]):
    def __init__(self, *assets: Asset,
                 engine: InferenceEngine = InferenceEngine.ONNX):
        super().__init__()

        if len(assets) == 0:
            assets = [MBLLENConfig.MBLLEN_Syn_LowLight_Noise_240x320]

        self.engine = InferenceEngineFactory.create(engine, assets,
                                                    flip_channels=True,
                                                    scale=255.0,
                                                    transpose=True,
                                                    padding=False)

        self.highpercent: int = 95  # should be in [85,100], linear amplification
        self.lowpercent: int = 5  # should be in [0,15], rescale the range [p%,1] to [0, 1]
        self.hsvgamma: int = 8  # should be in [6,10], increase the saturability
        self.maxrange: float = 8  # linear amplification range

    def setup(self):
        self.engine.setup()

    def process(self, image: np.ndarray) -> ImageResult:
        h, w = image.shape[:2]

        outputs = self.engine.process(image)
        output = outputs[self.engine.output_names[0]].squeeze()

        result = self._post_process(output, self.highpercent, self.lowpercent,
                                    self.hsvgamma / 10.0, self.maxrange / 10.0)

        result = cv2.resize(result, (w, h))
        return ImageResult(result)

    def release(self):
        self.engine.release()

    def _post_process(self, output: np.ndarray,
                      highpercent: float, lowpercent: float,
                      hsvgamma: float, maxrange: float) -> np.ndarray:
        gray_output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        percent_max = sum(sum(gray_output >= maxrange)) / sum(
            sum(gray_output <= 1.0))
        max_value = np.percentile(gray_output[:], highpercent)
        if percent_max < (100 - highpercent) / 100.:
            scale = maxrange / max_value
            output = output * scale
            output = np.minimum(output, 1.0)

        sub_value = np.percentile(gray_output[:], lowpercent)
        output = ((output - sub_value) * (1. / (1 - sub_value)))

        hsv_image = cv2.cvtColor(output, cv2.COLOR_RGB2HSV)
        h_value, s_value, v_value = cv2.split(hsv_image)
        s_value = np.power(s_value, hsvgamma)
        hsv_image = cv2.merge((h_value, s_value, v_value))
        output = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

        output = np.minimum(output, 1.0)
        output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
        return output

    def configure(self, args: Namespace):
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass

    @staticmethod
    def create(config: MBLLENConfig = MBLLENConfig.MBLLEN_Syn_LowLight_Noise_240x320) -> "MBLLENEstimator":
        model = config.value
        return MBLLENEstimator(model)
