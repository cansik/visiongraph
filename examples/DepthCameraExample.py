import argparse
from argparse import ArgumentParser

import cv2
import numpy as np

from visiongraph import add_logging_parameter
from visiongraph import vg
from visiongraph.input import add_input_step_choices


class DepthCameraExample(vg.BaseGraph):

    def __init__(self, input: vg.BaseDepthInput):
        super().__init__()
        self.input = input
        self.add_nodes(self.input)

    def _process(self):
        ts, frame = self.input.read()

        if frame is None:
            return

        depth_map = self.input.depth_map

        # combine color and depth frame
        if depth_map.shape != frame.shape:
            h, w = depth_map.shape[:2]
            frame = cv2.resize(frame, (w, h))

        output = np.hstack((self.input.depth_map, frame))

        cv2.imshow("Depth Input", output)
        if cv2.waitKey(15) & 0xFF == 27:
            self.close()

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass


def main():
    pipeline = DepthCameraExample(args.input())
    pipeline.configure(args)
    pipeline.open()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Depth Estimation Example", description="Example Pipeline")
    add_logging_parameter(parser)
    input_group = parser.add_argument_group("input provider")
    add_input_step_choices(input_group, default="realsense")

    args = parser.parse_args()
    args.depth = True

    main()
