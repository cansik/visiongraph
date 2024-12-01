from argparse import ArgumentParser
from dataclasses import dataclass
from typing import List, Callable, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult
from visiongraph.tracker.BaseObjectDetectionTracker import BaseObjectDetectionTracker
from visiongraph.util.VectorUtils import vector_as_list

CostFunctionType = Callable[[List[ObjectDetectionResult], List[ObjectDetectionResult]], np.ndarray]


@dataclass
class _FlateTrack:
    id: int
    reference: ObjectDetectionResult
    age: int = 0
    stale: int = 0

    def update_reference(self):
        self.reference.tracking_id = self.id


class FlateTracker(BaseObjectDetectionTracker):
    """
    Fast localization and tracking engine.
    """

    def __init__(self, max_cost: float = 0.2, min_alive: int = 0, max_lost: int = 5):
        """
        Initializes the FlateTracker with specified parameters.

        :param max_cost: Maximum cost for a trackable match.
        :param min_alive: Minimum number of frames a track must be visible to be considered alive.
        :param max_lost: Maximum number of frames a track can be lost before it is removed.
        """
        self.max_cost: float = max_cost

        self.min_alive: int = min_alive
        self.max_lost: int = max_lost

        self.include_stale: bool = False

        self.cost_function: Optional[CostFunctionType] = self._l2_cost_function

        self._tracks: List[_FlateTrack] = []
        self._unique_id: int = 0

    def setup(self):
        """
        Prepares the tracker for a new tracking session by clearing existing tracks and resetting the unique ID.
        """
        self._tracks.clear()
        self._unique_id = 0

    def _new_id(self) -> int:
        """
        Generates a new unique track ID.

        :return: A new unique track ID.
        """
        track_id = self._unique_id
        self._unique_id += 1
        return track_id

    def process(self, detections: List[ObjectDetectionResult]) -> ResultList[ObjectDetectionResult]:
        """
        Processes the given detections to update tracks and create new ones if necessary.

        :param detections: A list of detected objects to process.

        :return: A list of tracked objects.
        """
        # create cost matrix
        if len(self._tracks) == 0 or len(detections) == 0:
            cost_mat = np.zeros(shape=(0, 0), dtype=float)
        else:
            cost_mat = self.cost_function([t.reference for t in self._tracks], detections)

        row_indices, col_indices = linear_sum_assignment(cost_mat)
        row_indices = set(row_indices)

        # find all matches between tracks and detections
        matched_detections = set()
        index = 0
        for ti, track in enumerate(self._tracks):
            if ti in row_indices:
                # match has been found
                x = col_indices[index]
                score = cost_mat[ti, x]

                if score <= self.max_cost:
                    # match is valid
                    track.age += 1
                    track.stale = 0
                    track.reference = detections[x]
                    track.update_reference()
                    matched_detections.add(x)

                index += 1
                continue

            # track is lost
            track.stale += 1

        # process unmatched detections
        for di, detection in enumerate(detections):
            if di not in matched_detections:
                # new detection found
                track = _FlateTrack(self._new_id(), detection)
                track.update_reference()
                self._tracks.append(track)

        # clean up stale tracks
        self._tracks = [t for t in self._tracks if t.stale <= self.max_lost]

        return ResultList([t.reference for t in self._tracks
                           if t.age >= self.min_alive and (self.include_stale or t.stale == 0)])

    @staticmethod
    def _l2_cost_function(tracks: List[ObjectDetectionResult], detections: List[ObjectDetectionResult]) -> np.ndarray:
        """
        Computes the L2 cost matrix between tracks and detections based on their center positions.

        :param tracks: A list of tracked object detection results.
        :param detections: A list of detected object results.

        :return: The L2 cost matrix representing distances between tracks and detections.
        """
        track_centers = np.array([vector_as_list(h.bounding_box.center) for h in tracks], dtype=float)
        detection_centers = np.array([vector_as_list(h.bounding_box.center) for h in detections], dtype=float)

        distances = cdist(track_centers, detection_centers, metric="euclid")
        return distances

    @staticmethod
    def _iou_cost_function(tracks: List[ObjectDetectionResult], detections: List[ObjectDetectionResult]) -> np.ndarray:
        """
        Computes the Intersection over Union (IoU) cost matrix between tracks and detections.

        :param tracks: A list of tracked object detection results.
        :param detections: A list of detected object results.

        :return: The IoU cost matrix representing the overlap between tracks and detections.
        """
        cost_mat = np.zeros((len(tracks), len(detections)), dtype=float)

        for y, track in enumerate(tracks):
            for x, detection in enumerate(detections):
                cost_mat[y, x] = track.bounding_box.intersection_over_union(detection.bounding_box)

        return cost_mat

    def release(self):
        """
        Releases resources and clears all tracks.
        """
        self._tracks.clear()

    def configure(self, args):
        """
        Configures the tracker with parameters from the provided argument parser.

        :param args: Argument parser containing configuration parameters.
        """
        self.max_cost = self._get_param(args, "tracker_max_cost", self.max_cost)
        self.min_alive = self._get_param(args, "tracker_min_alive", self.min_alive)
        self.max_lost = self._get_param(args, "tracker_max_lost", self.max_lost)

    @staticmethod
    def add_params(parser: ArgumentParser):
        """
        Adds command line parameters for configuring the tracker.

        :param parser: The argument parser to add parameters to.
        """
        parser.add_argument("--tracker-max-cost", type=float, default=0.2, help="Max cost for trackable match.")
        parser.add_argument("--tracker-min-alive", type=int, default=0, help="Min frames trackable visible.")
        parser.add_argument("--tracker-max-lost", type=int, default=5, help="Max frames trackable not visible.")
