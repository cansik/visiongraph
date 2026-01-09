from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Dict

from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult
from visiongraph.tracker.BaseObjectDetectionTracker import BaseObjectDetectionTracker
from visiongraph.tracker.ObjectAssignmentSolver import ObjectAssignmentSolver, CostFunctionType


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

    def __init__(
        self,
        max_cost: float = 0.5,
        min_alive: int = 0,
        max_lost: int = 5,
        class_aware: bool = False,
        cost_function: Optional[CostFunctionType] = None,
    ):
        """
        Initializes the FlateTracker with specified parameters.

        :param max_cost: Maximum cost for a trackable match.
        :param min_alive: Minimum number of frames a track must be visible to be considered alive.
        :param max_lost: Maximum number of frames a track can be lost before it is removed.
        :param class_aware: If True, run class-aware matching.
        :param cost_function: Cost function to use for matching. Defaults to L2 distance.
        """
        self.max_cost: float = max_cost

        self.min_alive: int = min_alive
        self.max_lost: int = max_lost

        self.include_stale: bool = False

        self.class_aware: bool = class_aware
        self.cost_function: CostFunctionType = (
            cost_function if cost_function is not None else ObjectAssignmentSolver.l2_cost_function
        )

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
        if not self._tracks and not detections:
            return ResultList([])

        if not self._tracks:
            for det in detections:
                tr = _FlateTrack(self._new_id(), det)
                tr.update_reference()
                self._tracks.append(tr)
            return ResultList([t.reference for t in self._tracks])

        if not detections:
            for t in self._tracks:
                t.stale += 1
                t.reference.staleness = t.stale
            self._tracks = [t for t in self._tracks if t.stale <= self.max_lost]
            return ResultList(
                [t.reference for t in self._tracks if t.age >= self.min_alive and (self.include_stale or t.stale == 0)]
            )

        tracks_ref = [t.reference for t in self._tracks]

        solver = ObjectAssignmentSolver(self.cost_function, self.max_cost)

        assignments: Dict[ObjectDetectionResult, Optional[ObjectDetectionResult]] = {}
        unassigned_destinations: List[ObjectDetectionResult] = []

        if not self.class_aware:
            result = solver.solve(tracks_ref, detections)
            assignments = result.assignments
            unassigned_destinations = result.unassigned_destinations
        else:
            # Class-aware matching
            track_classes = [r.class_id for r in tracks_ref]
            det_classes = [d.class_id for d in detections]

            # Group by class
            tracks_by_class = defaultdict(list)
            for i, c in enumerate(track_classes):
                if c is not None:
                    tracks_by_class[c].append(tracks_ref[i])

            dets_by_class = defaultdict(list)
            for i, c in enumerate(det_classes):
                if c is not None:
                    dets_by_class[c].append(detections[i])

            known_classes = sorted(set(tracks_by_class.keys()) | set(dets_by_class.keys()))

            matched_tracks = set()
            matched_dets = set()

            # Step 1: per-class matching for known classes
            for cls_id in known_classes:
                t_subset = tracks_by_class.get(cls_id, [])
                d_subset = dets_by_class.get(cls_id, [])

                if not t_subset or not d_subset:
                    continue

                res = solver.solve(t_subset, d_subset)

                for src, dst in res.assignments.items():
                    assignments[src] = dst
                    matched_tracks.add(src)
                    if dst:
                        matched_dets.add(dst)

            # Step 2: handle unknown-class tracks against any remaining detections
            unknown_class_tracks = [
                t for i, t in enumerate(tracks_ref) if track_classes[i] is None and t not in matched_tracks
            ]
            rem_dets = [d for d in detections if d not in matched_dets]

            if unknown_class_tracks and rem_dets:
                res = solver.solve(unknown_class_tracks, rem_dets)
                for src, dst in res.assignments.items():
                    assignments[src] = dst
                    matched_tracks.add(src)
                    if dst:
                        matched_dets.add(dst)

            # Fill unassigned destinations
            unassigned_destinations = [d for d in detections if d not in matched_dets]

            # Ensure all tracks are in assignments
            for t in tracks_ref:
                if t not in assignments:
                    assignments[t] = None

        # Update tracks
        for track in self._tracks:
            dest = assignments.get(track.reference)

            if dest is not None:
                track.age += 1
                track.stale = 0
                track.reference = dest
                track.update_reference()
            else:
                track.stale += 1

            track.reference.staleness = track.stale

        # Create new tracks
        for det in unassigned_destinations:
            tr = _FlateTrack(self._new_id(), det)
            tr.update_reference()
            tr.reference.staleness = 0
            self._tracks.append(tr)

        # Clean up stale tracks
        self._tracks = [t for t in self._tracks if t.stale <= self.max_lost]

        return ResultList(
            [t.reference for t in self._tracks if t.age >= self.min_alive and (self.include_stale or t.stale == 0)]
        )

    def release(self):
        """
        Releases resources and clears all tracks.
        """
        self._tracks.clear()

    def configure(self, args):
        """
        Configures the tracker with parameters from the provided argument parser.
        """
        self.max_cost = self._get_param(args, "tracker_max_cost", self.max_cost)
        self.min_alive = self._get_param(args, "tracker_min_alive", self.min_alive)
        self.max_lost = self._get_param(args, "tracker_max_lost", self.max_lost)

    @staticmethod
    def add_params(parser: ArgumentParser):
        """
        Adds command line parameters for configuring the tracker.
        """
        parser.add_argument("--tracker-max-cost", type=float, default=0.5, help="Max cost for trackable match.")
        parser.add_argument("--tracker-min-alive", type=int, default=0, help="Min frames trackable visible.")
        parser.add_argument("--tracker-max-lost", type=int, default=5, help="Max frames trackable not visible.")
