import collections
from enum import Enum
from typing import Union, List

import numpy as np
from scipy.spatial.distance import cdist

from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.estimator.spatial.YOLOv5Detector import YOLOv5Detector
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.CrowdHumanResult import CrowdHumanResult
from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult
from visiongraph.tracker.ObjectAssignmentSolver import ObjectAssignmentSolver
from visiongraph.util.VectorUtils import vector_as_list, lerp_vector_2d


class CrowdHumanConfig(Enum):
    YOLOv5_N_640 = RepositoryAsset("crowdhuman-yolov5n-640.onnx")
    YOLOv5_N_P34_640 = RepositoryAsset("crowdhuman-yolov5n-p34-640.onnx")
    YOLOv5_N_P2_640 = RepositoryAsset("crowdhuman-yolov5n-p2-640.onnx")
    YOLOv5_S_640 = RepositoryAsset("crowdhuman-yolov5s-640.onnx")
    YOLOv5_S_P34_640 = RepositoryAsset("crowdhuman-yolov5s-p34-640.onnx")
    YOLOv5_S_P2_640 = RepositoryAsset("crowdhuman-yolov5s-p2-640.onnx")


class CrowdHumanDetector(YOLOv5Detector):

    def __init__(self, *assets: Asset, assign_head_to_person: bool = True):
        super().__init__(*assets, labels=["person", "head"], nms=True)

        self.assign_head_to_person = assign_head_to_person
        self.assignment_solver = ObjectAssignmentSolver(self.crowd_human_l2_cost_function)

    def process(self, image: np.ndarray) -> ResultList[Union[CrowdHumanResult, ObjectDetectionResult]]:
        results = super().process(image)

        if not self.assign_head_to_person:
            return results

        # split person and head
        objects = collections.defaultdict(list)
        for item in results:
            objects[item.class_name].append(item)

        people = objects["person"]
        heads = objects["head"]

        # associate heads and people
        assignment_result = self.assignment_solver.solve(people, heads)

        # post process results and check if head is inside person box
        output = ResultList[CrowdHumanResult]()
        for person, head in assignment_result.assignments.items():
            if head is None:
                continue

            if not head.bounding_box.contains(head.bounding_box.center):
                continue

            output.append(CrowdHumanResult(person, head))

        return output

    @staticmethod
    def create(config: CrowdHumanConfig = CrowdHumanConfig.YOLOv5_N_640) -> "CrowdHumanDetector":
        model = config.value
        return CrowdHumanDetector(model)

    @staticmethod
    def crowd_human_l2_cost_function(tracks: List[ObjectDetectionResult],
                                     detections: List[ObjectDetectionResult]) -> np.ndarray:

        def get_centers(results: List[ObjectDetectionResult]) -> np.ndarray:
            return np.array([vector_as_list(lerp_vector_2d(r.bounding_box.top_left, r.bounding_box.top_right, 0.5))
                             for r in results], dtype=float)

        track_centers = get_centers(tracks)
        detection_centers = get_centers(detections)

        distances = cdist(track_centers, detection_centers, metric="euclid")
        return distances
