from enum import Enum
from typing import List

from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.data.labels.COCO import COCO_80_LABELS
from visiongraph.estimator.openvino.OpenVinoObjectDetector import OpenVinoObjectDetector
from visiongraph.external.intel.adapters.openvino_adapter import OpenvinoAdapter, create_core
from visiongraph.external.intel.models.detection_model import DetectionModel
from visiongraph.external.intel.models.centernet import CenterNet


class CenterNetConfig(Enum):
    CenterNet_FP16 = (*RepositoryAsset.openVino("ctdet_coco_dlav0_512-fp16"), COCO_80_LABELS)
    CenterNet_FP32 = (*RepositoryAsset.openVino("ctdet_coco_dlav0_512-fp32"), COCO_80_LABELS)


class CenterNetDetector(OpenVinoObjectDetector):
    def __init__(self, model: Asset, weights: Asset, labels: List[str], min_score: float = 0.5, device: str = "CPU"):
        super().__init__(model, weights, labels, min_score, device)

    def _create_ie_model(self) -> DetectionModel:
        model_adapter = OpenvinoAdapter(create_core(), self.model.path, device=self.device)

        config = {
            'resize_type': None,
            'mean_values': None,
            'scale_values': None,
            'reverse_input_channels': True,
            'path_to_labels': None,
            'confidence_threshold': self.min_score,
            'input_size': None,  # The CTPN specific
            'num_classes': None,  # The NanoDet and NanoDetPlus specific
        }

        return CenterNet.create_model(CenterNet.__model__, model_adapter, config)

    @staticmethod
    def create(config: CenterNetConfig = CenterNetConfig.CenterNet_FP32) -> "CenterNetDetector":
        model, weights, labels = config.value
        return CenterNetDetector(model, weights, labels)
