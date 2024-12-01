from enum import Enum
from typing import Tuple, List, Optional, Set

import cv2
import numpy as np

from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.data.labels.COCO import COCO_80_LABELS
from visiongraph.estimator.engine.InferenceEngineFactory import InferenceEngine
from visiongraph.estimator.spatial.InstanceSegmentationEstimator import InstanceSegmentationEstimator
from visiongraph.estimator.spatial.UltralyticsYOLODetector import UltralyticsYOLODetector
from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.model.geometry.Size2D import Size2D
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.InstanceSegmentationResult import InstanceSegmentationResult
from visiongraph.util import ImageUtils, ResultUtils
from visiongraph.util.MathUtils import sigmoid


class YOLOv8SegmentationConfig(Enum):
    """
    Configuration options for YOLOv8 segmentation models.
    """
    YOLOv8_SEG_N = RepositoryAsset("yolov8n-seg.onnx"), COCO_80_LABELS
    YOLOv8_SEG_S = RepositoryAsset("yolov8s-seg.onnx"), COCO_80_LABELS
    YOLOv8_SEG_M = RepositoryAsset("yolov8m-seg.onnx"), COCO_80_LABELS
    YOLOv8_SEG_L = RepositoryAsset("yolov8l-seg.onnx"), COCO_80_LABELS
    YOLOv8_SEG_X = RepositoryAsset("yolov8x-seg.onnx"), COCO_80_LABELS


class YOLOv8SegmentationEstimator(UltralyticsYOLODetector[InstanceSegmentationResult], InstanceSegmentationEstimator):
    """
    YOLOv8 segmentation estimator for instance segmentation tasks.
    Inherits from UltralyticsYOLODetector and InstanceSegmentationEstimator.
    """

    def __init__(self, *assets: Asset, labels: List[str], min_score: float = 0.3, nms: bool = True,
                 nms_threshold: float = 0.5, nms_eta: Optional[float] = None, nms_top_k: Optional[int] = None,
                 engine: InferenceEngine = InferenceEngine.ONNX,
                 allowed_classes: Optional[Set[int]] = None, mask_threshold: float = 0.5):
        """
        Initializes the YOLOv8SegmentationEstimator.

        :param assets: The model assets.
        :param labels: The list of class labels.
        :param min_score: Minimum score for detections. Defaults to 0.3.
        :param nms: Whether to apply non-maximum suppression. Defaults to True.
        :param nms_threshold: Threshold for NMS. Defaults to 0.5.
        :param nms_eta: Eta parameter for NMS. Defaults to None.
        :param nms_top_k: Maximum number of boxes to keep after NMS. Defaults to None.
        :param engine: The inference engine to use. Defaults to InferenceEngine.ONNX.
        :param allowed_classes: Set of allowed class IDs. Defaults to None.
        :param mask_threshold: Threshold for mask predictions. Defaults to 0.5.
        """
        super().__init__(*assets, labels=labels, min_score=min_score, nms=nms, nms_threshold=nms_threshold,
                         nms_eta=nms_eta, nms_top_k=nms_top_k, engine=engine)

        self.allowed_classes: Optional[Set[int]] = allowed_classes
        self.mask_threshold = mask_threshold

    def _filter_predictions(self, predictions: np.ndarray, min_score: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filters predictions based on a minimum score.

        :param predictions: The predictions to filter.
        :param min_score: The minimum score threshold.

        :return: Filtered predictions and their corresponding scores.
        """
        predictions = predictions.T

        scores = np.max(predictions[:, 4:4 + len(self.labels)], axis=1)
        valid_indices = scores > self.min_score
        predictions = predictions[valid_indices, :]
        return predictions, scores[valid_indices]

    def process(self, image: np.ndarray) -> ResultList[InstanceSegmentationResult]:
        """
        Processes an image to perform instance segmentation.

        :param image: The input image.

        :return: The results of the instance segmentation.
        """
        ih, iw = image.shape[:2]
        output = self.engine.process(image)

        predictions = output[self.engine.output_names[0]]
        predictions = np.squeeze(predictions)

        # filter detection min score
        predictions, scores = self._filter_predictions(predictions, self.min_score)

        if len(scores) == 0:
            return ResultList()

        mask_output = output[self.engine.output_names[1]]
        mask_output: np.ndarray = np.squeeze(mask_output)

        boxes = predictions[:, :4]  # cxcywh
        class_ids = np.argmax(predictions[:, 4:4 + len(self.labels)], axis=1)
        mask_predictions = predictions[:, 4 + len(self.labels):]

        # prepare masks
        num_mask, mask_height, mask_width = mask_output.shape  # CHW
        masks = sigmoid(mask_predictions @ mask_output.reshape((num_mask, -1)))
        masks = masks.reshape((-1, mask_height, mask_width))

        padding_box = output.padding_box.scale(1 / output.image_size.width, 1 / output.image_size.height)

        h, w = self.engine.first_input_shape[2:]

        blur_size = (int(iw / mask_width), int(ih / mask_height))

        results = ResultList()
        for box, score, class_id, mask in zip(boxes, scores, class_ids, masks):

            # filter classes if necessary
            if self.allowed_classes is not None:
                if class_id not in self.allowed_classes:
                    continue

            # process bounding box
            wh = box[2:4]
            xy = box[0:2]
            xy -= wh * 0.5
            bbox = BoundingBox2D(xy[0], xy[1], wh[0], wh[1]).scale(1 / w, 1 / h)

            # process mask
            region = self._crop(bbox.to_array(tl_br_format=True), mask.shape)
            cropped = np.zeros(mask.shape, dtype=mask.dtype)
            cropped[region] = mask[region]

            mask_box = padding_box.scale(mask.shape[1], mask.shape[0])
            cropped = ImageUtils.roi(cropped, mask_box)

            cropped = cv2.resize(cropped, (iw, ih), interpolation=cv2.INTER_CUBIC)
            cropped = cv2.blur(cropped, blur_size)

            cropped = (cropped > self.mask_threshold).astype(np.uint8)

            result = InstanceSegmentationResult(class_id, self.labels[class_id], score, cropped, bbox)
            result.map_coordinates(Size2D.from_image(mask), (iw, ih), src_roi=mask_box)
            results.append(result)

        if self.nms:
            results = ResultList(ResultUtils.non_maximum_suppression(results, self.min_score, self.nms_threshold))
        return results

    @staticmethod
    def _crop(bbox, shape):
        """
        Crops a bounding box to fit within the given shape.

        :param bbox: The bounding box coordinates.
        :param shape: The shape of the area to crop.

        :return: Slices for the cropped area.
        """
        x1 = int(max(bbox[0] * shape[1], 0))
        y1 = int(max(bbox[1] * shape[0], 0))
        x2 = int(max(bbox[2] * shape[1], 0))
        y2 = int(max(bbox[3] * shape[0], 0))
        return slice(y1, y2), slice(x1, x2)

    @staticmethod
    def create(
            config: YOLOv8SegmentationConfig = YOLOv8SegmentationConfig.YOLOv8_SEG_S) -> "YOLOv8SegmentationEstimator":
        """
        Creates an instance of YOLOv8SegmentationEstimator using the specified configuration.

        :param config: The configuration for the estimator. Defaults to YOLOv8SegmentationConfig.YOLOv8_SEG_S.

        :return: An instance of the YOLOv8SegmentationEstimator.
        """
        model, labels = config.value
        return YOLOv8SegmentationEstimator(model, labels=labels)
