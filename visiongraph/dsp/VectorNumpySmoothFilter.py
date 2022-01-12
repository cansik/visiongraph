from argparse import ArgumentParser, Namespace
from typing import TypeVar

import numpy as np
from vector import VectorNumpy

from visiongraph.GraphNode import GraphNode
from visiongraph.dsp.BaseFilterNumpy import BaseFilterNumpy
from visiongraph.dsp.OneEuroFilterNumpy import OneEuroFilterNumpy
from visiongraph.util.VectorUtils import vector_to_array, array_to_vector

InputType = TypeVar('InputType', bound=VectorNumpy)
OutputType = TypeVar('OutputType', bound=VectorNumpy)


class VectorNumpySmoothFilter(GraphNode[InputType, OutputType]):

    def __init__(self, np_filter: BaseFilterNumpy = OneEuroFilterNumpy(np.zeros(1))):
        self._filter = np_filter

    def setup(self):
        pass

    def process(self, data: InputType) -> OutputType:
        array = vector_to_array(data)
        result = self._filter(array)
        return array_to_vector(result)

    def release(self):
        pass

    def configure(self, args: Namespace):
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass
