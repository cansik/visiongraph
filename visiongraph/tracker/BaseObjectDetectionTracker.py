from abc import abstractmethod, ABC
from typing import List

from visiongraph.GraphNode import GraphNode
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult


class BaseObjectDetectionTracker(GraphNode[ResultList[ObjectDetectionResult], ResultList[ObjectDetectionResult]], ABC):

    @abstractmethod
    def process(self, data: List[ObjectDetectionResult]) -> ResultList[ObjectDetectionResult]:
        pass
