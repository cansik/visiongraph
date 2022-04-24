import argparse
from argparse import ArgumentParser

import cv2
import numpy as np

from visiongraph.BaseGraph import BaseGraph
from visiongraph.estimator.spatial.camera.ArUcoCameraPoseEstimator import ArUcoCameraPoseEstimator
from visiongraph.input import add_input_step_choices
from visiongraph.input.BaseInput import BaseInput
from visiongraph.util.LoggingUtils import add_logging_parameter


class CameraPoseEstimationExample(BaseGraph):

    def __init__(self, input: BaseInput):
        super().__init__()
        self.input = input

        self.network = ArUcoCameraPoseEstimator(
            camera_matrix=np.array([[356.92661953, 0., 221.39625447],
                                    [0., 549.91986118, 377.42600875],
                                    [0., 0., 1.]], dtype=float),
            fisheye_distortion=np.array([5.50597053e+00, -1.92784557e+01, 6.93713729e-01,
                                         1.99492438e-02, 3.38608778e+01], dtype=float)
        )

        self.add_nodes(self.input, self.network)

    def _process(self):
        ts, frame = self.input.read()

        if frame is None:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        result = self.network.process(gray)
        result.annotate(frame)

        cv2.putText(frame, f"Distance: {result.position.mag:.2f}",
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

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
