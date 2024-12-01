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
        """
        Initializes the LandmarkEmbedder node with a given embedding function.

        :param embedding_function: A function that takes a landmark detection result and returns an optional numpy array representing the embedding.
        """
        self.embedding_function = embedding_function

    def setup(self):
        """
        Sets up the LandmarkEmbedder node. This method is called before processing any input data.
        """
        pass

    def process(self, detections: ResultList[T]) -> ResultList[LandmarkEmbeddingResult]:
        """
        Processes a list of landmark detection results and computes their embeddings using the provided embedding function.

        :param detections: A list of landmark detection results.

        :return: A list of landmark embedding results.
        """
        results = ResultList()
        for detection in detections:
            embedding = self.embedding_function(detection)

            if embedding is None:
                continue

            results.append(LandmarkEmbeddingResult(embedding, detection))

        return results

    def release(self):
        """
        Releases any resources used by the LandmarkEmbedder node.
        """
        pass

    def configure(self, args: Namespace):
        """
        Configures the LandmarkEmbedder node based on the provided command-line arguments.

        :param args: A namespace containing command-line arguments.
        """
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        """
        Adds parameters to the argparse parser for the LandmarkEmbedder node.
        """
        pass
