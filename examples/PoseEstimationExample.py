import argparse
from argparse import ArgumentParser

import cv2

from visiongraph.estimator.embedding.LandmarkEmbedder import LandmarkEmbedder
from visiongraph.estimator.spatial.SlidingWindowEstimator import SlidingWindowEstimator
from visiongraph.BaseGraph import BaseGraph
from visiongraph.estimator.spatial.pose import add_pose_estimation_step_choices
from visiongraph.estimator.spatial.pose.PoseEstimator import PoseEstimator
from visiongraph.input import add_input_step_choices
from visiongraph.input.BaseInput import BaseInput
from visiongraph.tracker.CentroidTracker import CentroidTracker
from visiongraph.tracker.MotpyTracker import MotpyTracker
from visiongraph.util.LoggingUtils import add_logging_parameter, setup_logging
from visiongraph.util.PoseUtils import embed_pose
from visiongraph.util.TimeUtils import FPSTracer


class PoseEstimationExample(BaseGraph):

    def __init__(self, input: BaseInput, pose_network: PoseEstimator, sliding_window: bool = False):
        super().__init__()
        self.input = input

        if sliding_window:
            self.network = SlidingWindowEstimator(
                pose_network, 128, (256, 256), 0.5
            )
        else:
            self.network = pose_network
        self.fps_tracer = FPSTracer()
        self.tracker = MotpyTracker()

        self.embedder = LandmarkEmbedder(embed_pose)

        self.add_nodes(self.input, self.network, self.tracker, self.embedder)

    def _process(self):
        ts, frame = self.input.read()

        if frame is None:
            return

        results = self.network.process(frame)
        results = self.tracker.process(results)

        embeddings = self.embedder.process(results)

        for result in results:
            result.annotate(frame, min_score=0.1)

        self.fps_tracer.update()
        if not args.performance:
            cv2.putText(frame, "FPS: %.0f" % self.fps_tracer.smooth_fps,
                        (7, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("Pose Estimator", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                self.close()
        else:
            print("\033[K", end='')
            print("FPS: %.0f" % self.fps_tracer.smooth_fps)

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
    parser.add_argument("-p", "--performance", action="store_true", help="Enable performance test (no UI).")

    input_group = parser.add_argument_group("input provider")
    add_input_step_choices(input_group)

    pose_group = parser.add_argument_group("pose estimator")
    add_pose_estimation_step_choices(pose_group)

    PoseEstimationExample.add_params(parser)

    args = parser.parse_args()

    main()
