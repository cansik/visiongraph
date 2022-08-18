import argparse
from argparse import ArgumentParser

import cv2

from visiongraph import current_millis
from visiongraph.BaseGraph import BaseGraph
from visiongraph.estimator.spatial.camera.CameraChessboardCalibrator import CameraChessboardCalibrator
from visiongraph.input import add_input_step_choices
from visiongraph.input.BaseInput import BaseInput
from visiongraph.util.LoggingUtils import add_logging_parameter


class CameraCalibratorTool(BaseGraph):

    def __init__(self, input: BaseInput):
        super().__init__()
        self.input = input

        self.max_samples = 30
        self.network = CameraChessboardCalibrator(max_samples=self.max_samples)

        self.wait_time = 1000
        self.last_ts = 0

        self.add_nodes(self.input, self.network)

    def _process(self):
        ts, frame = self.input.read()

        if frame is None:
            return

        if current_millis() - self.last_ts > self.wait_time:
            self.last_ts = current_millis()
            result = self.network.process(frame)

            if result is not None:
                intrinsics = result.intrinsics

                print("Intrinsics Matrix:")
                print(intrinsics.intrinsic_matrix)

                print()
                print("Distortion Coefficients:")
                print(intrinsics.distortion_coefficients)

                intrinsics.save("media/calibration.json")

                self.close()

        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Samples: {len(self.network.imgpoints)} / {self.max_samples}",
                    (7, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

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
    parser = argparse.ArgumentParser("Camera Calibrator Tool", description="Example Pipeline")
    add_logging_parameter(parser)
    input_group = parser.add_argument_group("input provider")
    add_input_step_choices(input_group)

    args = parser.parse_args()

    main()
