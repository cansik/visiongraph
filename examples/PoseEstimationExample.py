import argparse
from argparse import ArgumentParser

import cv2

from visiongraph.BaseGraph import BaseGraph
from visiongraph.estimator.spatial.pose import add_pose_estimation_step_choices
from visiongraph.estimator.spatial.pose.PoseEstimator import PoseEstimator
from visiongraph.input import add_input_step_choices
from visiongraph.input.BaseInput import BaseInput
from visiongraph.tracker.CentroidTracker import CentroidTracker
from visiongraph.util.LoggingUtils import add_logging_parameter
from visiongraph.util.TimeUtils import FPSTracer


class PoseEstimationExample(BaseGraph):

    def __init__(self, input: BaseInput, pose_network: PoseEstimator):
        super().__init__()
        self.input = input
        self.network = pose_network
        self.fps_tracer = FPSTracer()
        self.tracker = CentroidTracker()

        self.add_nodes(self.input, self.network, self.tracker)

    def _process(self):
        ts, frame = self.input.read()

        if frame is None:
            return

        results = self.network.process(frame)
        results = self.tracker.track(results)

        for result in results:
            result.annotate(frame)

        self.fps_tracer.update()
        if not args.performance:
            cv2.putText(frame, "FPS: %.0f" % self.fps_tracer.smooth_fps,
                        (7, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

            cv2.imshow("Pose Estimator", frame)
            if cv2.waitKey(15) & 0xFF == 27:
                self.close()
        else:
            print("\033[K", end='')
            print("FPS: %.0f" % self.fps_tracer.smooth_fps)

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass


def main():
    pipeline = PoseEstimationExample(args.input(), args.pose_estimator())
    pipeline.configure(args)
    pipeline.open()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Pose Estimation Example", description="Example Pipeline")
    add_logging_parameter(parser)
    parser.add_argument("-p", "--performance", action="store_true", help="Enable performance test (no UI).")

    input_group = parser.add_argument_group("input provider")
    add_input_step_choices(input_group)

    pose_group = parser.add_argument_group("pose estimator")
    add_pose_estimation_step_choices(pose_group)

    CentroidTracker.add_params(parser)

    args = parser.parse_args()

    main()
