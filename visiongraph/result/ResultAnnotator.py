from argparse import ArgumentParser, Namespace
from typing import List, Optional

import numpy as np

from visiongraph.GraphNode import GraphNode
from visiongraph.result.BaseResult import BaseResult
from visiongraph.result.ResultDict import ResultDict, DEFAULT_IMAGE_KEY


class ResultAnnotator(GraphNode[ResultDict, np.ndarray]):
    """
    A graph node responsible for annotating the results of a computation on an image.
    """

    def __init__(self, image_key: str = DEFAULT_IMAGE_KEY, result_keys: Optional[List[str]] = None, **annotation_args):
        """
        Initializes the ResultAnnotator node.

        :param image_key: The key of the image in the result dictionary. Defaults to DEFAULT_IMAGE_KEY.
        :param result_keys: A list of keys of the results to be annotated. Defaults to None.
        :param annotation_args: Keyword arguments used for annotations.
        """
        self.image_key = image_key
        self.result_keys = result_keys
        self.annotation_args = annotation_args

    def setup(self):
        """
        Sets up the node before processing data.
        """
        pass

    def process(self, data: ResultDict) -> ResultDict:
        """
        Annotates the results of a computation on an image.

        :param data: The result dictionary to be processed.

        :return: The annotated result dictionary.
        """
        image = data[self.image_key]

        for key, result in data.items():
            if self.result_keys is not None:
                if key not in self.result_keys:
                    continue

            if isinstance(result, BaseResult):
                result.annotate(image, **self.annotation_args)

        return data

    def release(self):
        """
        Releases resources after processing data.
        """
        pass

    def configure(self, args: Namespace):
        """
        Configures the node based on the provided arguments.

        :param args: The command-line arguments.
        """
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        """
        Adds parameters to the parser for configuring this graph node.

        :param parser: The argument parser to be configured.
        """
        pass
