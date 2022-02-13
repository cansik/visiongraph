import time
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
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self.filters: Dict[int, VectorNumpySmoothFilter] = {}

    def setup(self):
        pass

    def process(self, data: InputType) -> OutputType:
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
        pass

    def configure(self, args: Namespace):
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass
