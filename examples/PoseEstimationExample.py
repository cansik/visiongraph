import argparse
from argparse import ArgumentParser

import cv2

from visiongraph.BaseGraph import BaseGraph
from visiongraph.estimator.spatial.pose import add_pose_estimation_step_choices
from visiongraph.estimator.spatial.pose.PoseEstimator import PoseEstimator
from visiongraph.input import add_input_step_choices
from visiongraph.input.BaseInput import BaseInput
from visiongraph.util.LoggingUtils import add_logging_parameter


class PoseEstimationExample(BaseGraph):

    def __init__(self, input: BaseInput, pose_network: PoseEstimator):
        super().__init__()
        self.input = input
        self.network = pose_network

        self.add_nodes(self.input, self.network)

    def _process(self):
        ts, frame = self.input.read()

        if frame is None:
            return

        results = self.network.estimate(frame)
        for result in results:
            result.annotate(frame)

        cv2.imshow("Pose Estimator", frame)
        if cv2.waitKey(15) & 0xFF == 27:
            self.close()

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass


def main():
    pipeline = PoseEstimationExample(args.input(),  args.pose_estimator())
    pipeline.configure(args)
    pipeline.open()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Pose Estimation Example", description="Example Pipeline")
    add_logging_parameter(parser)

    input_group = parser.add_argument_group("input provider")
    add_input_step_choices(input_group)

    pose_group = parser.add_argument_group("pose estimator")
    add_pose_estimation_step_choices(pose_group)

    args = parser.parse_args()

    main()
