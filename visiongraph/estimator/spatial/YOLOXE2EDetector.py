from typing import List, Optional

import numpy as np

from visiongraph.data.Asset import Asset
from visiongraph.estimator.engine.InferenceEngineFactory import InferenceEngine, InferenceEngineFactory
from visiongraph.estimator.spatial.ObjectDetector import ObjectDetector
from visiongraph.model.NMSOptions import NMSOptions
from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.model.geometry.Size2D import Size2D
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult
from visiongraph.util.ResultUtils import non_maximum_suppression_from_options


class YOLOXE2EDetector(ObjectDetector):
    """
    YOLOXE2EDetector class represents an object detector using YOLO-X vision engine.
    """

    def __init__(
        self,
        *assets: Asset,
        labels: List[str],
        min_score: float = 0.3,
        nms_options: Optional[NMSOptions] = None,
        engine: InferenceEngine = InferenceEngine.ONNX,
    ):
        """
        Initialize the YOLOXE2EDetector object with specified parameters.

        :param *assets: Variable length argument list of assets.
        :param labels: List of label names.
        :param min_score: Minimum score for detected objects (default is 0.3).
        :param nms_options: Non-Maximum-Suppression options.
        :param engine: Inference engine to be used (default is InferenceEngine.ONNX).
        """
        super().__init__(min_score)
        self.engine = InferenceEngineFactory.create(engine, assets, flip_channels=True, scale=255.0, padding=True)
        # set padding color
        self.engine.padding_color = (125, 125, 125)

        self.labels: List[str] = labels
        self.nms_options = nms_options or NMSOptions()

    def setup(self):
        """
        Setup the inference engine before processing.
        """
        self.engine.setup()

    def process(self, image: np.ndarray) -> ResultList[ObjectDetectionResult]:
        """
        Process the input image to detect objects and return the results.

        :param image: Input image to be processed.

        :return: List of object detection results.
        """
        output = self.engine.process(image)
        boxes = output["boxes"]
        labels = output["labels"]

        boxes = boxes[0]
        labels = labels[0]

        h, w = self.engine.first_input_shape[2:]

        # filter detection min score
        output_indices = np.where(boxes[:, 4] > self.min_score)[0]

        # create result list
        results = ResultList()
        for i in output_indices:
            label = labels[i]
            if label < 0:
                continue

            box = boxes[i]

            x1, y1, x2, y2, score = box

            # find label
            label_index = label

            # process bounding box
            bbox = BoundingBox2D(x1, y1, x2 - x1, y2 - y1).scale(1 / w, 1 / h)

            detection = ObjectDetectionResult(label_index, self.labels[label_index], score, bbox)
            detection.map_coordinates(output.image_size, Size2D.from_image(image), src_roi=output.padding_box)
            results.append(detection)

        if self.nms_options.enabled:
            results = ResultList(non_maximum_suppression_from_options(results, self.nms_options))
        return results

    def release(self):
        """
        Release any resources held by the inference engine.
        """
        self.engine.release()
