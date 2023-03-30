import argparse
import faulthandler
from argparse import ArgumentParser

import cv2
import numpy as np

from visiongraph.estimator.spatial.camera.ChArUcoCalibrator import ChArUcoCalibrator
from visiongraph.estimator.spatial.camera.ChessboardCalibrator import ChessboardCalibrator
from visiongraph.util.ArgUtils import add_step_choice_argument
from visiongraph.util.TimeUtils import current_millis
from visiongraph.BaseGraph import BaseGraph
from visiongraph.estimator.spatial.camera.BoardCameraCalibrator import BoardCameraCalibrator
from visiongraph.input import add_input_step_choices
from visiongraph.input.BaseInput import BaseInput
from visiongraph.util.LoggingUtils import add_logging_parameter

BOARD_CALIBRATORS = {
    "chessboard": ChessboardCalibrator,
    # "charuco": ChArUcoCalibrator
}


class CameraCalibratorTool(BaseGraph):

    def __init__(self, input: BaseInput, calibrator: BoardCameraCalibrator):
        super().__init__()
        self.input = input

        self.calibrator = calibrator
        self.output_path = "media/calibration.json"

        self.wait_time = 1000
        self.last_ts = 0

        self._success_border_counter = 0

        self.add_nodes(self.input, self.calibrator)

    def _process(self):
        ts, frame = self.input.read()

        if frame is None:
            return

        if current_millis() - self.last_ts > self.wait_time:
            self.last_ts = current_millis()
            result = self.calibrator.process(frame)

            if self.calibrator.board_detected:
                self._success_border_counter = 5

            if result is not None:
                intrinsics = result.intrinsics

                np.set_printoptions(suppress=True)

                print("Intrinsics Matrix:")
                print(intrinsics.intrinsic_matrix)

                print()
                print("Distortion Coefficients:")
                print(intrinsics.distortion_coefficients)

                np.set_printoptions(suppress=False)

                intrinsics.save(self.output_path)

                self.close()

        frame = cv2.flip(frame, 1)

        if self._success_border_counter > 0:
            self._success_border_counter -= 1

            h, w = frame.shape[:2]
            frame = cv2.rectangle(frame, (0, 0), (w - 1, h - 1), [0, 255, 0], 8)

        cv2.putText(frame, f"Samples: {self.calibrator.sample_count} / {self.calibrator.max_samples}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("Camera Calibrator", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            self.close()

    def configure(self, args: argparse.Namespace):
        super().configure(args)

        c, r = args.board_size
        self.calibrator.rows = r
        self.calibrator.columns = c
        self.calibrator.max_samples = args.max_samples
        self.wait_time = int(args.wait_time)
        self.output_path = args.calibration

    @staticmethod
    def add_params(parser: ArgumentParser):
        parser.add_argument("--board-size", type=int, nargs=2, required=True,
                            metavar=("columns", "rows"), help="Calibration board size.")
        parser.add_argument("--max-samples", type=int, default=30, help="How many calibration samples are gathered.")
        parser.add_argument("--wait-time", type=int, default=1000, help="How long to wait between capture (ms).")
        parser.add_argument("--calibration", type=str, default="calibration.json",
                            help="Path where the calibration is stored.")


def main():
    parser = argparse.ArgumentParser("Camera Calibrator Tool", description="Calibrate cameras with boards.")
    add_logging_parameter(parser)
    input_group = parser.add_argument_group("input provider")
    add_input_step_choices(input_group)

    add_step_choice_argument(parser, BOARD_CALIBRATORS, "--calibrator", help="Board calibrator system.",
                             default="chessboard", add_params=True)
    CameraCalibratorTool.add_params(parser)

    args = parser.parse_args()

    pipeline = CameraCalibratorTool(args.input(), args.calibrator(6, 7))
    pipeline.configure(args)
    pipeline.open()


if __name__ == "__main__":
    faulthandler.enable()
    main()
