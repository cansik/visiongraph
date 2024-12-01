from abc import abstractmethod, ABC
from typing import List

from visiongraph.GraphNode import GraphNode
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult


class BaseObjectDetectionTracker(GraphNode[ResultList[ObjectDetectionResult], ResultList[ObjectDetectionResult]], ABC):
    """
    Abstract base class for object detection trackers.

    This class serves as a starting point for all object detection trackers. It defines the interface that must be implemented by any concrete tracker.
    """

    @abstractmethod
    def process(self, data: List[ObjectDetectionResult]) -> ResultList[ObjectDetectionResult]:
        """
        Processes a list of detected objects.

        :param data: The list of objects to be processed.

        :return: A list of results containing the processed object detection information.
        """
        pass
