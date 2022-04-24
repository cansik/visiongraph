import argparse
from argparse import ArgumentParser

import cv2

from visiongraph.BaseGraph import BaseGraph
from visiongraph.estimator.spatial.camera.CameraChessboardCalibrator import CameraChessboardCalibrator
from visiongraph.input import add_input_step_choices
from visiongraph.input.BaseInput import BaseInput
from visiongraph.util.LoggingUtils import add_logging_parameter


class CameraCalibratorTool(BaseGraph):

    def __init__(self, input: BaseInput):
        super().__init__()
        self.input = input

        self.network = CameraChessboardCalibrator(max_samples=100)

        self.add_nodes(self.input, self.network)

    def _process(self):
        ts, frame = self.input.read()

        if frame is None:
            return

        result = self.network.process(frame)

        if result is not None:
            print("Intrinsics Matrix:")
            print(result.intrinsic_matrix)

            print()
            print("Distortion Coefficients:")
            print(result.distortion_coefficients)

            self.close()

        cv2.imshow("Camera Pose Example", frame)
        if cv2.waitKey(15) & 0xFF == 27:
            self.close()

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass


def main():
    pipeline = CameraCalibratorTool(args.input())
    pipeline.configure(args)
    pipeline.open()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Camera Pose Example", description="Example Pipeline")
    add_logging_parameter(parser)
    input_group = parser.add_argument_group("input provider")
    add_input_step_choices(input_group)

    args = parser.parse_args()

    main()
