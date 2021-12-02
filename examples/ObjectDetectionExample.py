import argparse
from argparse import ArgumentParser

import cv2

from visiongraph.Pipeline import Pipeline
from visiongraph.estimator.spatial.SSDDetector import SSDDetector
from visiongraph.estimator.spatial.YOLODetector import YOLODetector, YOLOConfig
from visiongraph.input import add_input_step_choices
from visiongraph.input.BaseInput import BaseInput
from visiongraph.tracker.ObjectDetectionTracker import ObjectDetectionTracker
from visiongraph.util.LoggingUtils import add_logging_parameter


class ObjectDetectionExample(Pipeline):

    def __init__(self, input: BaseInput, multi_threaded: bool = True, deamon: bool = True):
        super().__init__(multi_threaded, deamon)
        self.input = input
        self.network = SSDDetector.create() # YOLODetector.create()
        self.tracker = ObjectDetectionTracker()

        self.add_nodes(self.input, self.network, self.tracker)

    def _process(self):
        ts, frame = self.input.read()

        if frame is None:
            return

        results = self.network.estimate(frame)
        results = self.tracker.track(results)

        for result in results:
            result.annotate(frame)

        cv2.imshow("Object Detection", frame)
        cv2.waitKey(15)

    @staticmethod
    def add_params(parser: ArgumentParser):
        ObjectDetectionTracker.add_params(parser)


def main():
    pipeline = ObjectDetectionExample(args.input(), multi_threaded=False)
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
