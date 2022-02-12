from argparse import ArgumentParser
from typing import Optional, List

from visiongraph.GraphNode import GraphNode
from visiongraph.external.motpy import MultiObjectTracker, Detection
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult


class MotpyTracker(GraphNode[ResultList[ObjectDetectionResult], ResultList[ObjectDetectionResult]]):
    def __init__(self, delta_time: float = 1.0 / 10.0, min_steps_alive: int = 0):
        self.delta_time = delta_time
        self.min_steps_alive = min_steps_alive

        self.tracker: Optional[MultiObjectTracker] = None

    def setup(self):
        if self.tracker is None:
            self.tracker = MultiObjectTracker(dt=self.delta_time)

    def process(self, data: List[ObjectDetectionResult]) -> ResultList[ObjectDetectionResult]:
        detections = [Detection(box=d.bounding_box.to_array(tl_br_format=True), reference=d)
                      for d in data]
        self.tracker.step(detections)
        active_tracks = self.tracker.active_tracks(min_steps_alive=self.min_steps_alive)

        results = ResultList()
        for track in active_tracks:
            detection = track.reference
            detection.tracking_id = track.id
            results.append(detection)

        return results

    def release(self):
        pass

    def configure(self, args):
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass
