from argparse import ArgumentParser
from typing import Optional, List

import numpy as np

from visiongraph.external.motrackers.Tracker import Tracker
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult
from visiongraph.tracker.BaseObjectDetectionTracker import BaseObjectDetectionTracker


class CentroidTracker(BaseObjectDetectionTracker):
    """
    A class to track objects in a video sequence using centroid detection.
    """

    def __init__(self, tracker: Optional[Tracker] = None):
        """
        Initializes a new instance of the CentroidTracker class.

        :param tracker: The underlying object tracker. Defaults to None.
        """
        self.tracker = tracker
        self.enabled = True
        self.max_lost = 0

    def setup(self):
        """
        Sets up the tracker by creating a new instance if one does not already exist.

        """
        if not self.tracker:
            self.tracker = Tracker(max_lost=self.max_lost, tracker_output_format='raw')

    def track(self, detections: List[ObjectDetectionResult]) -> ResultList[ObjectDetectionResult]:
        """
        Tracks the given detections using the underlying object tracker.

        :param detections: The list of detection results to be tracked.

        :return: The list of tracked detection results.
        """
        if not self.enabled:
            return detections

        inputs = [(list(d.bounding_box), d.score, d.tracking_id, i) for i, d in enumerate(detections)]
        bboxes, scores, ids, references = zip(*inputs) if len(detections) else ([], [], [], [])
        tracks = self.tracker.update(np.asarray(bboxes), np.asarray(scores), np.asarray(ids), np.asarray(references))

        tracked_detections = ResultList()
        for track in tracks:
            detection = detections[track.reference]
            detection.tracking_id = track.id
            tracked_detections.append(detection)

        return tracked_detections

    def process(self, data: List[ObjectDetectionResult]) -> ResultList[ObjectDetectionResult]:
        """
        Processes the given data by tracking the detections.

        :param data: The list of detection results to be processed.

        :return: The list of tracked detection results.
        """
        return self.track(data)

    def release(self):
        """
        Releases the underlying object tracker.

        """
        self.tracker = None

    def configure(self, args):
        """
        Configures the tracker based on the given command-line arguments.

        :param args: The command-line arguments to be used for configuration.
        """
        self.max_lost = self._get_param(args, "tracker_max_lost", self.max_lost)

    @staticmethod
    def add_params(parser: ArgumentParser):
        """
        Adds parameters to the parser that control the behavior of the CentroidTracker.

        :param parser: The parser to which the parameters will be added.
        """
        parser.add_argument("--tracker-max-lost", type=int, default=5, help="Max frames trackable not visible.")
