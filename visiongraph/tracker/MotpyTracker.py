from argparse import ArgumentParser
from typing import Optional, List

from visiongraph.GraphNode import GraphNode
from visiongraph.external.motpy import MultiObjectTracker, Detection
from visiongraph.external.motpy.tracker import EPS
from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult


class MotpyTracker(GraphNode[ResultList[ObjectDetectionResult], ResultList[ObjectDetectionResult]]):
    def __init__(self, delta_time: float = 1.0 / 10.0,
                 min_iou: float = 0.1,
                 multi_match_min_iou: float = 1. + EPS,
                 min_steps_alive: int = -1,
                 max_staleness_to_positive_ratio: float = 3.0,
                 max_staleness: float = 12.0,
                 use_predicted_bounding_box: bool = False):

        self.delta_time = delta_time
        self.min_steps_alive = min_steps_alive
        self.min_iou = min_iou
        self.multi_match_min_iou = multi_match_min_iou
        self.max_staleness_to_positive_ratio = max_staleness_to_positive_ratio
        self.max_staleness = max_staleness

        self.use_predicted_bounding_box = use_predicted_bounding_box

        self.tracker: Optional[MultiObjectTracker] = None

    def setup(self):
        if self.tracker is None:
            self.tracker = MultiObjectTracker(dt=self.delta_time,
                                              tracker_kwargs={'max_staleness': self.max_staleness},
                                              matching_fn_kwargs={'min_iou': self.min_iou,
                                                                  'multi_match_min_iou': self.multi_match_min_iou})

    def process(self, data: List[ObjectDetectionResult]) -> ResultList[ObjectDetectionResult]:
        detections = [Detection(box=d.bounding_box.to_array(tl_br_format=True), reference=d)
                      for d in data]
        self.tracker.step(detections)
        active_tracks = self.tracker.active_tracks(min_steps_alive=self.min_steps_alive)

        results = ResultList()
        for track in active_tracks:
            detection: ObjectDetectionResult = track.reference
            detection.tracking_id = track.id
            detection.staleness = track.staleness

            if self.use_predicted_bounding_box:
                detection.box = BoundingBox2D.from_array(track.box, tl_br_format=True)

            results.append(detection)

        return results

    def release(self):
        self.tracker = None

    def configure(self, args):
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass
