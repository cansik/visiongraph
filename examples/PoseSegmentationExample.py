import argparse
from argparse import ArgumentParser

import cv2

from visiongraph.BaseGraph import BaseGraph
from visiongraph.estimator.spatial.pose import MediaPipePoseEstimator
from visiongraph.input import add_input_step_choices
from visiongraph.input.BaseInput import BaseInput
from visiongraph.util.LoggingUtils import add_logging_parameter
from visiongraph.util.TimeUtils import FPSTracer


class PoseSegmentationExample(BaseGraph):
    def __init__(self, input: BaseInput):
        super().__init__()
        self.input = input
        self.network = MediaPipePoseEstimator.create()
        self.network.enable_segmentation = True

        self.fps_tracer = FPSTracer()

        self.add_nodes(self.input, self.network)

    def _process(self):
        ts, frame = self.input.read()

        if frame is None:
            return

        results = self.network.process(frame)

        for result in results:
            result.annotate(frame)

        self.fps_tracer.update()
        cv2.putText(
            frame,
            "FPS: %.0f" % self.fps_tracer.smooth_fps,
            (7, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Pose Estimator", frame)
        if cv2.waitKey(15) & 0xFF == 27:
            self.close()

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass


def main():
    pipeline = PoseSegmentationExample(args.input())
    pipeline.configure(args)
    pipeline.open()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Pose Segmentation Example", description="Example Pipeline")
    add_logging_parameter(parser)
    parser.add_argument("-p", "--performance", action="store_true", help="Enable performance test (no UI).")

    input_group = parser.add_argument_group("input provider")
    add_input_step_choices(input_group)

    args = parser.parse_args()

    main()
