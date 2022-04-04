from enum import Enum
from typing import List

from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.data.labels.COCO import COCO_80_LABELS
from visiongraph.estimator.openvino.OpenVinoObjectDetector import OpenVinoObjectDetector
from visiongraph.external.intel.adapters.openvino_adapter import OpenvinoAdapter, create_core
from visiongraph.external.intel.models.detection_model import DetectionModel
from visiongraph.external.intel.models.detr import DETR


class DETRConfig(Enum):
    DETR_Resnet50_FP16 = (*RepositoryAsset.openVino("detr-resnet50-fp16"), COCO_80_LABELS)
    DETR_Resnet50_FP32 = (*RepositoryAsset.openVino("detr-resnet50-fp32"), COCO_80_LABELS)


class DETRDetector(OpenVinoObjectDetector):
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

        return DETR.create_model(DETR.__model__, model_adapter, config)

    @staticmethod
    def create(config: DETRConfig = DETRConfig.DETR_Resnet50_FP32) -> "DETRDetector":
        model, weights, labels = config.value
        return DETRDetector(model, weights, labels)

    def _get_label(self, index: int):
        return super()._get_label(index - 1)