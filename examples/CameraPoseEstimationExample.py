import argparse
from argparse import ArgumentParser

import cv2

from visiongraph.BaseGraph import BaseGraph
from visiongraph.estimator.spatial.camera.ArUcoCameraPoseEstimator import ArUcoCameraPoseEstimator
from visiongraph.input import add_input_step_choices
from visiongraph.input.BaseInput import BaseInput
from visiongraph.model.CameraIntrinsics import CameraIntrinsics
from visiongraph.util.LoggingUtils import add_logging_parameter


class CameraPoseEstimationExample(BaseGraph):
    def __init__(self, input: BaseInput):
        super().__init__()
        self.input = input

        intrinsics = CameraIntrinsics.load("media/calibration.json")

        self.network = ArUcoCameraPoseEstimator(
            camera_matrix=intrinsics.intrinsic_matrix, fisheye_distortion=intrinsics.distortion_coefficients
        )

        self.add_nodes(self.input, self.network)

    def _process(self):
        ts, frame = self.input.read()

        if frame is None:
            return

        # gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        result = self.network.process(frame)

        if result is not None:
            result.annotate(frame)
            cv2.putText(
                frame, f"Distance: {result.position.mag:.2f}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )

        cv2.imshow("Camera Pose Example", frame)
        if cv2.waitKey(15) & 0xFF == 27:
            self.close()

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass


def main():
    pipeline = CameraPoseEstimationExample(args.input())
    pipeline.configure(args)
    pipeline.open()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Camera Pose Example", description="Example Pipeline")
    add_logging_parameter(parser)
    input_group = parser.add_argument_group("input provider")
    add_input_step_choices(input_group)

    args = parser.parse_args()

    main()
