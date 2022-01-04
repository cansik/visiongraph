from argparse import ArgumentParser
from typing import Optional, List, Any

import numpy as np

from visiongraph.model.chain.ChainableNode import ChainableNode
from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult
from visiongraph.tracker.motrackers.Tracker import Tracker


class ObjectDetectionTracker(ChainableNode):
    def __init__(self, tracker: Optional[Tracker] = None):
        self.tracker = tracker
        self.enabled = True
        self.max_lost = 0

    def setup(self):
        if not self.tracker:
            self.tracker = Tracker(max_lost=self.max_lost, tracker_output_format='raw')

    def track(self, detections: List[ObjectDetectionResult]) -> List[ObjectDetectionResult]:
        if not self.enabled:
            return detections

        inputs = [(list(d.bounding_box), d.score, d.tracking_id, i) for i, d in enumerate(detections)]
        bboxes, scores, ids, references = zip(*inputs) if len(detections) else ([], [], [], [])
        tracks = self.tracker.update(np.asarray(bboxes), np.asarray(scores), np.asarray(ids), np.asarray(references))

        tracked_detections = []
        for track in tracks:
            detection = detections[track.reference]
            detection.tracking_id = track.id
            tracked_detections.append(detection)

        return tracked_detections

    def release(self):
        pass

    def _chain_apply(self, detections, *args, **kwargs) -> Any:
        return self.track(detections)

    def configure(self, args):
        self.max_lost = args.tracker_max_lost

    @staticmethod
    def add_params(parser: ArgumentParser):
        parser.add_argument("--tracker-max-lost", type=int, default=5, help="Max frames trackable not visible.")
