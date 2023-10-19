from argparse import ArgumentParser, Namespace
from typing import TypeVar, Callable, Optional

import numpy as np

from visiongraph.GraphNode import GraphNode
from visiongraph.result.LandmarkEmbeddingResult import LandmarkEmbeddingResult
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.LandmarkDetectionResult import LandmarkDetectionResult

T = TypeVar("T", bound=LandmarkDetectionResult)


class LandmarkEmbedder(GraphNode[ResultList[T], ResultList[LandmarkEmbeddingResult]]):

    def __init__(self, embedding_function: Callable[[T], Optional[np.ndarray]]):
        self.embedding_function = embedding_function

    def setup(self):
        pass

    def process(self, detections: ResultList[T]) -> ResultList[LandmarkEmbeddingResult]:
        results = ResultList()
        for detection in detections:
            embedding = self.embedding_function(detection)

            if embedding is None:
                continue

            results.append(LandmarkEmbeddingResult(embedding, detection))

        return results

    def release(self):
        pass

    def configure(self, args: Namespace):
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass
