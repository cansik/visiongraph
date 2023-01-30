import argparse
from argparse import ArgumentParser

import cv2

from visiongraph.BaseGraph import BaseGraph
from visiongraph.estimator.spatial.SSDDetector import SSDDetector, SSDConfig
from visiongraph.estimator.spatial.SlidingWindowEstimator import SlidingWindowEstimator
from visiongraph.input import add_input_step_choices
from visiongraph.input.BaseInput import BaseInput
from visiongraph.tracker.CentroidTracker import CentroidTracker
from visiongraph.tracker.FlateTracker import FlateTracker
from visiongraph.tracker.MotpyTracker import MotpyTracker
from visiongraph.util.LoggingUtils import add_logging_parameter, setup_logging


class ObjectDetectionExample(BaseGraph):

    def __init__(self, input: BaseInput, sliding_window=False):
        super().__init__()
        self.input = input
        self.network = SSDDetector.create()

        if sliding_window:
            self.network = SlidingWindowEstimator(
                SSDDetector.create(SSDConfig.PersonDetection_0200_256x256_FP32), 128, (256, 256), 0.8
            )

        self.tracker = FlateTracker()

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
        if cv2.waitKey(1) & 0xFF == 27:
            self.close()

    @staticmethod
    def add_params(parser: ArgumentParser):
        CentroidTracker.add_params(parser)
        parser.add_argument("--sliding-window", action="store_true", help="Use a sliding window for detection.")


def main():
    pipeline = ObjectDetectionExample(args.input(), sliding_window=args.sliding_window)
    pipeline.configure(args)
    pipeline.open()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Object Detection Example", description="Example Object Detection Pipeline")
    add_logging_parameter(parser)
    input_group = parser.add_argument_group("input provider")
    add_input_step_choices(input_group)
    ObjectDetectionExample.add_params(parser)

    args = parser.parse_args()

    setup_logging(args.loglevel)

    main()
