from enum import Enum
from typing import Optional, List

import cv2
import numpy as np

from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.data.labels.COCO import COCO_80_LABELS
from visiongraph.estimator.openvino.VisionInferenceEngine import VisionInferenceEngine
from visiongraph.estimator.spatial.InstanceSegmentationEstimator import InstanceSegmentationEstimator
from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.InstanceSegmentationResult import InstanceSegmentationResult

_IS_NAME = "instance-segmentation-security"


class MaskRCNNConfig(Enum):
    # 0002 = https://docs.openvino.ai/2021.4/omz_models_model_instance_segmentation_security_0002.html
    ResNet50_1024x768_INT8 = (*RepositoryAsset.openVino(f"{_IS_NAME}-0002-fp16-int8"), COCO_80_LABELS)
    ResNet50_1024x768_FP16 = (*RepositoryAsset.openVino(f"{_IS_NAME}-0002-fp16"), COCO_80_LABELS)
    ResNet50_1024x768_FP32 = (*RepositoryAsset.openVino(f"{_IS_NAME}-0002-fp32"), COCO_80_LABELS)

    # 0091 = https://docs.openvino.ai/2021.4/omz_models_model_instance_segmentation_security_0091.html
    ResNet101_1344x800_INT8 = (*RepositoryAsset.openVino(f"{_IS_NAME}-0091-fp16-int8"), COCO_80_LABELS)
    ResNet101_1344x800_FP16 = (*RepositoryAsset.openVino(f"{_IS_NAME}-0091-fp16"), COCO_80_LABELS)
    ResNet101_1344x800_FP32 = (*RepositoryAsset.openVino(f"{_IS_NAME}-0091-fp32"), COCO_80_LABELS)

    # 0228 = https://docs.openvino.ai/2021.4/omz_models_model_instance_segmentation_security_0228.html
    ResNet101_608_INT8 = (*RepositoryAsset.openVino(f"{_IS_NAME}-0228-fp16-int8"), COCO_80_LABELS)
    ResNet101_608_FP16 = (*RepositoryAsset.openVino(f"{_IS_NAME}-0228-fp16"), COCO_80_LABELS)
    ResNet101_608_FP32 = (*RepositoryAsset.openVino(f"{_IS_NAME}-0228-fp32"), COCO_80_LABELS)

    # 1039 = https://docs.openvino.ai/2021.4/omz_models_model_instance_segmentation_security_1039.html
    EfficientNet_480_INT8 = (*RepositoryAsset.openVino(f"{_IS_NAME}-1039-fp16-int8"), COCO_80_LABELS)
    EfficientNet_480_FP16 = (*RepositoryAsset.openVino(f"{_IS_NAME}-1039-fp16"), COCO_80_LABELS)
    EfficientNet_480_FP32 = (*RepositoryAsset.openVino(f"{_IS_NAME}-1039-fp32"), COCO_80_LABELS)

    # 1040 = https://docs.openvino.ai/2021.4/omz_models_model_instance_segmentation_security_1040.html
    EfficientNet_608_INT8 = (*RepositoryAsset.openVino(f"{_IS_NAME}-1040-fp16-int8"), COCO_80_LABELS)
    EfficientNet_608_FP16 = (*RepositoryAsset.openVino(f"{_IS_NAME}-1040-fp16"), COCO_80_LABELS)
    EfficientNet_608_FP32 = (*RepositoryAsset.openVino(f"{_IS_NAME}-1040-fp32"), COCO_80_LABELS)


class MaskRCNNEstimator(InstanceSegmentationEstimator[InstanceSegmentationResult]):
    def __init__(self, model: Asset, weights: Asset, labels: List[str],
                 min_score: float = 0.5, device: str = "AUTO"):
        super().__init__(min_score)
        self.model = model
        self.weights = weights

        self.width: Optional[int] = None
        self.height: Optional[int] = None

        self.labels = labels

        self.device = device
        self.engine: Optional[VisionInferenceEngine] = None

    def setup(self):
        self.engine = VisionInferenceEngine(self.model, self.weights,
                                            flip_channels=True, device=self.device)
        self.engine.setup()
        _, _, self.height, self.width = self.engine.first_input_shape

    def process(self, data: np.ndarray) -> ResultList[InstanceSegmentationResult]:
        h, w = data.shape[:2]

        scale_x = self.width / w
        scale_y = self.height / h

        outputs = self.engine.process(data)
        raw_results = self._postprocess(outputs, scale_x, scale_y, h, w, self.width, self.height, self.min_score)

        results = ResultList()
        for i in range(len(raw_results[0])):
            score = float(raw_results[0][i])
            class_id = int(raw_results[1][i]) - 1
            bbox = raw_results[2][i]
            mask = raw_results[3][i]

            box = BoundingBox2D(bbox[0] / w, bbox[1] / h, (bbox[2] - bbox[0]) / w, (bbox[3] - bbox[1]) / h)
            res = InstanceSegmentationResult(class_id, self.labels[class_id], score, mask, box)
            results.append(res)

        return results

    def release(self):
        self.engine.release()

    @staticmethod
    def _postprocess(outputs, scale_x, scale_y, frame_height,
                     frame_width, input_height, input_width, conf_threshold):
        segmentoly_postprocess = 'raw_masks' in outputs
        boxes = outputs['boxes'] if segmentoly_postprocess else outputs['boxes'][:, :4]
        scores = outputs['scores'] if segmentoly_postprocess else outputs['boxes'][:, 4]
        boxes[:, 0::2] /= scale_x
        boxes[:, 1::2] /= scale_y
        if segmentoly_postprocess:
            classes = outputs['classes'].astype(np.uint32)
        else:
            classes = outputs['labels'].astype(np.uint32) + 1
        masks = []
        masks_name = 'raw_masks' if segmentoly_postprocess else 'masks'
        for box, cls, raw_mask in zip(boxes, classes, outputs[masks_name]):
            raw_cls_mask = raw_mask[cls, ...] if segmentoly_postprocess else raw_mask
            mask = MaskRCNNEstimator._segm_postprocess(box, raw_cls_mask, frame_height, frame_width)
            masks.append(mask)
        # Filter out detections with low confidence.
        detections_filter = scores > conf_threshold
        scores = scores[detections_filter]
        classes = classes[detections_filter]
        boxes = boxes[detections_filter]
        masks = [segm for segm, is_valid in zip(masks, detections_filter) if is_valid]
        return scores, classes, boxes, masks

    @staticmethod
    def _expand_box(box, scale):
        w_half = (box[2] - box[0]) * .5
        h_half = (box[3] - box[1]) * .5
        x_c = (box[2] + box[0]) * .5
        y_c = (box[3] + box[1]) * .5
        w_half *= scale
        h_half *= scale
        box_exp = np.zeros(box.shape)
        box_exp[0] = x_c - w_half
        box_exp[2] = x_c + w_half
        box_exp[1] = y_c - h_half
        box_exp[3] = y_c + h_half
        return box_exp

    @staticmethod
    def _segm_postprocess(box, raw_cls_mask, im_h, im_w):
        # Add zero border to prevent upsampling artifacts on segment borders.
        raw_cls_mask = np.pad(raw_cls_mask, ((1, 1), (1, 1)), 'constant', constant_values=0)
        extended_box = MaskRCNNEstimator._expand_box(box,
                                                     raw_cls_mask.shape[0] / (raw_cls_mask.shape[0] - 2.0)).astype(int)
        w, h = np.maximum(extended_box[2:] - extended_box[:2] + 1, 1)
        x0, y0 = np.clip(extended_box[:2], a_min=0, a_max=[im_w, im_h])
        x1, y1 = np.clip(extended_box[2:] + 1, a_min=0, a_max=[im_w, im_h])

        raw_cls_mask = cv2.resize(raw_cls_mask, (w, h)) > 0.5
        mask = raw_cls_mask.astype(np.uint8)
        # Put an object mask in an image mask.
        im_mask = np.zeros((im_h, im_w), dtype=np.uint8)
        im_mask[y0:y1, x0:x1] = mask[(y0 - extended_box[1]):(y1 - extended_box[1]),
                                (x0 - extended_box[0]):(x1 - extended_box[0])]
        return im_mask

    @staticmethod
    def create(config: MaskRCNNConfig = MaskRCNNConfig.EfficientNet_480_FP32) -> "MaskRCNNEstimator":
        model, weights, labels = config.value
        return MaskRCNNEstimator(model, weights, labels)
