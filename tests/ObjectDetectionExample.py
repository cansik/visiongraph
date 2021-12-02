import argparse
from argparse import ArgumentParser

import cv2

from visiongraph.Pipeline import Pipeline
from visiongraph.estimator.spatial.YOLODetector import YOLODetector, YOLOConfig
from visiongraph.input import add_input_step_choices
from visiongraph.input.BaseInput import BaseInput
from visiongraph.util.LoggingUtils import add_logging_parameter


class ObjectDetectionExample(Pipeline):

    def __init__(self, input: BaseInput, multi_threaded: bool = True, deamon: bool = True):
        super().__init__(multi_threaded, deamon)
        self.input = input
        self.object_detector = YOLODetector.create()

        self.add_nodes(self.input, self.object_detector)

    def _process(self):
        ts, frame = self.input.read()

        if frame is None:
            return

        results = self.object_detector.estimate(frame)
        for result in results:
            result.annotate(frame)

        cv2.imshow("Object Detection", frame)
        cv2.waitKey(15)

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass


def main():
    pipeline = ObjectDetectionExample(args.input(), multi_threaded=False)
    pipeline.configure(args)
    pipeline.open()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Object Detection Example", description="Example Object Detection Pipeline")
    add_logging_parameter(parser)
    input_group = parser.add_argument_group("input provider")
    add_input_step_choices(input_group)

    args = parser.parse_args()

    main()
