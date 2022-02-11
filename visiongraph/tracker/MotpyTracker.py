import logging
from argparse import ArgumentParser
from typing import Optional, List, Dict

from visiongraph.external.motpy import MultiObjectTracker, Detection, ModelPreset

from visiongraph.GraphNode import GraphNode
from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult


class MotpyTracker(GraphNode[ResultList[ObjectDetectionResult], ResultList[ObjectDetectionResult]]):
    def __init__(self, delta_time: float = 1.0 / 10.0, min_steps_alive: int = 0):
        self.delta_time = delta_time
        self.min_steps_alive = min_steps_alive

        self.tracker: Optional[MultiObjectTracker] = None

        self._next_tracking_id = 0
        self._track_lut: Dict[str, int] = {}

    def setup(self):
        if self.tracker is None:
            self.tracker = MultiObjectTracker(dt=self.delta_time)

    def process(self, data: List[ObjectDetectionResult]) -> ResultList[ObjectDetectionResult]:
        detections = [Detection(box=d.bounding_box.to_array(tl_br_format=True))
                      for i, d in enumerate(data)]
        self.tracker.step(detections)
        active_tracks = self.tracker.active_tracks(min_steps_alive=self.min_steps_alive)

        # instead of object detections, update input data
        print(self.tracker.matches)

        # update tracks lookup tables
        current_track_lut = {}
        for track in active_tracks:
            if track.id in self._track_lut:
                current_track_lut[track.id] = self._track_lut[track.id]
            else:
                current_track_lut[track.id] = self._next_tracking_id
                self._next_tracking_id += 1
        self._track_lut = current_track_lut

        results = ResultList()
        for track in active_tracks:
            res = ObjectDetectionResult(track.class_id, track.class_id,
                                        track.score, BoundingBox2D.from_array(track.box, tl_br_format=True))
            res.tracking_id = self._track_lut[track.id]
            results.append(res)

        return results

    def release(self):
        pass

    def configure(self, args):
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass
