import argparse
from argparse import ArgumentParser

import cv2

from visiongraph.BaseGraph import BaseGraph
from visiongraph.estimator.spatial.SSDDetector import SSDDetector
from visiongraph.input import add_input_step_choices
from visiongraph.input.BaseInput import BaseInput
from visiongraph.tracker.CentroidTracker import CentroidTracker
from visiongraph.util.LoggingUtils import add_logging_parameter


class ObjectDetectionExample(BaseGraph):

    def __init__(self, input: BaseInput):
        super().__init__()
        self.input = input
        self.network = SSDDetector.create()
        self.tracker = CentroidTracker()

        self.add_nodes(self.input, self.network, self.tracker)

    def _process(self):
        ts, frame = self.input.read()

        if frame is None:
            return

        results = self.network.process(frame)
        results = self.tracker.process(results)

        for result in results:
            result.annotate(frame)

        cv2.imshow("Object Detection", frame)
        if cv2.waitKey(15) & 0xFF == 27:
            self.close()

    @staticmethod
    def add_params(parser: ArgumentParser):
        CentroidTracker.add_params(parser)


def main():
    pipeline = ObjectDetectionExample(args.input())
    pipeline.configure(args)
    pipeline.open()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Object Detection Example", description="Example Object Detection Pipeline")
    add_logging_parameter(parser)
    input_group = parser.add_argument_group("input provider")
    add_input_step_choices(input_group)
    ObjectDetectionExample.add_params(parser)

    args = parser.parse_args()

    main()
