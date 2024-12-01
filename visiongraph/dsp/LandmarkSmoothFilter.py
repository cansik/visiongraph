from argparse import ArgumentParser, Namespace
from typing import TypeVar, List, Dict

from visiongraph.GraphNode import GraphNode
from visiongraph.dsp.OneEuroFilterNumpy import OneEuroFilterNumpy
from visiongraph.dsp.VectorNumpySmoothFilter import VectorNumpySmoothFilter
from visiongraph.result.spatial.LandmarkDetectionResult import LandmarkDetectionResult
from visiongraph.util.VectorUtils import vector_to_array

InputType = TypeVar('InputType', bound=List[LandmarkDetectionResult])
OutputType = TypeVar('OutputType', bound=List[LandmarkDetectionResult])


class LandmarkSmoothFilter(GraphNode[InputType, OutputType]):
    """
    A class to smooth landmark detections using OneEuro filter and VectorNumpySmoothFilter.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        """
        Initializes the LandmarkSmoothFilter object with default parameters.

        :param min_cutoff: The minimum cutoff value for the OneEuro filter.
        :param beta: The parameter value for the OneEuro filter.
        :param d_cutoff: The cutoff value for the VectorNumpySmoothFilter.
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self.filters: Dict[int, VectorNumpySmoothFilter] = {}

    def setup(self):
        """
        Sets up the filters by creating a new OneEuroFilterNumpy and VectorNumpySmoothFilter for each tracking id.
        """

    def process(self, data: InputType) -> OutputType:
        """
        Smooths the landmark detections in the input data using the stored filters.

        :param data: The input data containing landmark detections.

        :return: The smoothed landmark detections.
        """
        for detection in data:
            # smoothing only works on tracked landmark detections
            if detection.tracking_id < 0:
                continue

            # create or get filter
            if detection.tracking_id not in self.filters:
                landmarks = vector_to_array(detection.landmarks)
                smooth_filter = VectorNumpySmoothFilter(OneEuroFilterNumpy(x0=landmarks,
                                                                           min_cutoff=self.min_cutoff,
                                                                           beta=self.beta,
                                                                           d_cutoff=self.d_cutoff,
                                                                           invalid_value=0.0))
                smooth_filter.setup()
                self.filters.update({detection.tracking_id: smooth_filter})

            smooth_filter = self.filters[detection.tracking_id]

            detection.landmarks = smooth_filter.process(detection.landmarks)

        # remove dead filters
        indices = {det.tracking_id for det in data}
        dead_ids = self.filters.keys() - indices
        for index in dead_ids:
            self.filters.pop(index)

        return data

    def release(self):
        """
        Releases the resources used by the LandmarkSmoothFilter object.
        """

    def configure(self, args: Namespace):
        """
        Configures the LandmarkSmoothFilter object based on the provided command-line arguments.

        :param args: The command-line arguments passed to the LandmarkSmoothFilter constructor.
        """

    @staticmethod
    def add_params(parser: ArgumentParser):
        """
        Adds parameters for the LandmarkSmoothFilter class to the provided argument parser.

        :param parser: The argument parser used to define the command-line interface.
        """
