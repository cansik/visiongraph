from argparse import ArgumentParser, Namespace
from typing import List, Optional

import numpy as np

from visiongraph.GraphNode import GraphNode
from visiongraph.result.BaseResult import BaseResult
from visiongraph.result.ResultDict import ResultDict, DEFAULT_IMAGE_KEY


class ResultAnnotator(GraphNode[ResultDict, np.ndarray]):
    def __init__(self, image_key: str = DEFAULT_IMAGE_KEY, result_keys: Optional[List[str]] = None, **annotation_args):
        self.image_key = image_key
        self.result_keys = result_keys
        self.annotation_args = annotation_args

    def setup(self):
        pass

    def process(self, data: ResultDict) -> ResultDict:
        image = data[self.image_key]

        for key, result in data.items():
            if self.result_keys is not None:
                if key not in self.result_keys:
                    continue

            if isinstance(result, BaseResult):
                result.annotate(image, **self.annotation_args)

        return data

    def release(self):
        pass

    def configure(self, args: Namespace):
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass
