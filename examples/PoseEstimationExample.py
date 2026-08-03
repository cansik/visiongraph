import argparse
from argparse import ArgumentParser

import cv2

from visiongraph.BaseGraph import BaseGraph
from visiongraph.estimator.spatial.pose import add_pose_estimation_step_choices
from visiongraph.estimator.spatial.pose.PoseEstimator import PoseEstimator
from visiongraph.estimator.spatial.SlidingWindowEstimator import SlidingWindowEstimator
from visiongraph.input import add_input_step_choices
from visiongraph.input.BaseInput import BaseInput
from visiongraph.tracker.CentroidTracker import CentroidTracker
from visiongraph.tracker.MotpyTracker import MotpyTracker
from visiongraph.util.LoggingUtils import add_logging_parameter, setup_logging
from visiongraph.util.TimeUtils import FPSTracer


class PoseEstimationExample(BaseGraph):
    def __init__(self, input: BaseInput, pose_network: PoseEstimator, sliding_window: bool = False):
        super().__init__()
        self.input = input

        if sliding_window:
            self.network = SlidingWindowEstimator(pose_network, 128, (256, 256), 0.5)
        else:
            self.network = pose_network
        self.fps_tracer = FPSTracer()
        self.tracker = MotpyTracker()

        self.add_nodes(self.input, self.network, self.tracker)

    def _process(self):
        _, frame = self.input.read()

        if frame is None:
            return

        results = self.network.process(frame)
        results = self.tracker.process(results)

        for result in results:
            result.annotate(frame, min_score=0.1)

        self.fps_tracer.update()
        cv2.putText(
            frame,
            f"FPS: {self.fps_tracer.smooth_fps:.0f}",
            (7, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Pose Estimator", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            self.close()

    @staticmethod
    def add_params(parser: ArgumentParser):
        CentroidTracker.add_params(parser)
        parser.add_argument("--sliding-window", action="store_true", help="Use a sliding window for detection.")


def main():
    setup_logging(args.loglevel)

    pipeline = PoseEstimationExample(args.input(), args.pose_estimator(), sliding_window=args.sliding_window)
    pipeline.configure(args)
    pipeline.open()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Pose Estimation Example", description="Example Pipeline")
    add_logging_parameter(parser)

    input_group = parser.add_argument_group("input provider")
    add_input_step_choices(input_group)

    pose_group = parser.add_argument_group("pose estimator")
    add_pose_estimation_step_choices(pose_group)

    PoseEstimationExample.add_params(parser)

    args = parser.parse_args()

    main()
