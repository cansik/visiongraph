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
    """
    A filter node that applies a smoothing operation to the input data using a one-euro filter.
    """

    def __init__(self, np_filter: BaseFilterNumpy = OneEuroFilterNumpy(np.zeros(1))):
        """
        Initializes the VectorNumpySmoothFilter node.

        :param np_filter: The filter to be used for smoothing. Defaults to an instance of OneEuroFilterNumpy with a zero-filled input.
        """
        self._filter = np_filter

    def setup(self):
        """
        Sets up the necessary components and configurations for the node to operate.
        """
        pass

    def process(self, data: InputType) -> OutputType:
        """
        Applies the smoothing operation to the input data.

        :param data: The input data to be smoothed.

        :return: The smoothed output data as a VectorNumpy instance.
        """
        array = vector_to_array(data)
        result = self._filter(array)
        return array_to_vector(result)

    def release(self):
        """
        Releases any resources allocated by the node.
        """
        pass

    def configure(self, args: Namespace):
        """
        Configures the node based on the provided command-line arguments.

        :param args: The parsed command-line arguments.
        """
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        """
        Adds parameters to the parser for configuring the VectorNumpySmoothFilter node.

        :param parser: The argument parser instance.
        """
        pass
