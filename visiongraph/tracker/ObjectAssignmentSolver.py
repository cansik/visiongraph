import sys
from dataclasses import dataclass
from typing import Optional, Callable, List, TypeVar, Generic, Dict

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult
from visiongraph.util.VectorUtils import vector_as_list

CostFunctionType = Callable[[List[ObjectDetectionResult], List[ObjectDetectionResult]], np.ndarray]

T = TypeVar("T", bound=ObjectDetectionResult)


@dataclass
class ObjectAssignmentResult(Generic[T]):
    """
    Data class representing the result of an object assignment process.
    """
    assignments: Dict[T, Optional[T]]
    unassigned_destinations: List[T]
    costs: Dict[T, float]

    @property
    def unassigned_sources(self) -> List[T]:
        """
        Returns a list of source objects that were not assigned to any destination.

        :return: A list of unassigned source objects.
        """
        return [k for k, v in self.assignments.items() if v is None]


class ObjectAssignmentSolver(Generic[T]):
    """
    Solver for assigning source objects to destination objects based on a given cost function.
    """

    def __init__(self, cost_function: Optional[CostFunctionType] = None, max_cost: float = sys.maxsize):
        """
        Initializes the ObjectAssignmentSolver with a given cost function and maximum cost.

        :param cost_function: The cost function to use for assignment. Defaults to L2 cost function.
        :param max_cost: The maximum allowable cost for an assignment to be considered valid. Defaults to sys.maxsize.
        """
        self.cost_function: CostFunctionType = self.l2_cost_function if cost_function is None else cost_function
        self.max_cost = max_cost

    def solve(self, source_list: List[T], destination_list: List[T]) -> ObjectAssignmentResult[T]:
        """
        Solves the assignment problem for the given lists of source and destination objects.

        :param source_list: The list of source objects.
        :param destination_list: The list of destination objects.

        :return: The result of the assignment process.
        """
        # create cost matrix
        if len(source_list) == 0 or len(destination_list) == 0:
            cost_mat = np.zeros(shape=(0, 0), dtype=float)
        else:
            cost_mat = self.cost_function(source_list, destination_list)

        row_indices, col_indices = linear_sum_assignment(cost_mat)
        row_indices = set(row_indices)

        # find all matches between tracks and detections
        assignments = dict()
        costs = dict()
        matched_detections = set()
        index = 0
        for ti, source in enumerate(source_list):
            if ti in row_indices:
                # match has been found
                x = col_indices[index]
                cost = cost_mat[ti, x]

                if cost <= self.max_cost:
                    # match is valid
                    dest = destination_list[x]
                    assignments[source] = dest
                    costs[source] = float(cost)
                    matched_detections.add(x)
                else:
                    assignments[source] = None
                    costs[source] = self.max_cost

                index += 1
                continue

            # no association
            assignments[source] = None
            costs[source] = self.max_cost

        # process unmatched detections
        unassigned_destinations = []
        for di, dest in enumerate(destination_list):
            if di not in matched_detections:
                unassigned_destinations.append(dest)

        return ObjectAssignmentResult(assignments, unassigned_destinations, costs)

    @staticmethod
    def l2_cost_function(tracks: List[T], detections: List[T]) -> np.ndarray:
        """
        Computes the L2 (Euclidean) distance cost matrix between source and destination objects.

        :param tracks: The list of source objects.
        :param detections: The list of destination objects.

        :return: The cost matrix based on L2 distances.
        """
        track_centers = np.array([vector_as_list(h.bounding_box.center) for h in tracks], dtype=float)
        detection_centers = np.array([vector_as_list(h.bounding_box.center) for h in detections], dtype=float)

        distances = cdist(track_centers, detection_centers, metric="euclid")
        return distances

    @staticmethod
    def iou_cost_function(tracks: List[T], detections: List[T], class_exclusive: bool = False) -> np.ndarray:
        """
        Computes the Intersection-over-Union (IoU) cost matrix between source and destination objects.

        :param tracks: The list of source objects.
        :param detections: The list of destination objects.
        :param class_exclusive: Whether to only consider matches between objects of the same class. Defaults to False.

        :return: The cost matrix based on IoU values.
        """
        cost_mat = np.zeros((len(tracks), len(detections)), dtype=float)

        for y, track in enumerate(tracks):
            for x, detection in enumerate(detections):
                if class_exclusive and track.class_id != detection.class_id:
                    cost_mat[y, x] = sys.maxsize
                    continue

                cost_mat[y, x] = 1.0 - track.bounding_box.intersection_over_union(detection.bounding_box)
        return cost_mat
