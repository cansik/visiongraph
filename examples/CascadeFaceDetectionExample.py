import argparse
from argparse import ArgumentParser

import cv2

from visiongraph.Pipeline import Pipeline
from visiongraph.estimator.spatial.SpatialCascadeEstimator import SpatialCascadeEstimator
from visiongraph.estimator.spatial.face.AdasFaceDetector import AdasFaceDetector
from visiongraph.estimator.spatial.face.OpenVinoFaceDetector import OpenVinoFaceDetector
from visiongraph.estimator.spatial.face.landmark.RegressionLandmarkEstimator import RegressionLandmarkEstimator
from visiongraph.estimator.spatial.face.pose.AdasHeadPoseEstimator import AdasHeadPoseEstimator
from visiongraph.input import add_input_step_choices
from visiongraph.input.BaseInput import BaseInput
from visiongraph.util.LoggingUtils import add_logging_parameter


class CascadeFaceDetectionExample(Pipeline):

    def __init__(self, input: BaseInput, multi_threaded: bool = True, deamon: bool = True):
        super().__init__(multi_threaded, deamon)
        self.input = input
        self.network = SpatialCascadeEstimator(AdasFaceDetector.create(),
                                               landmarks=RegressionLandmarkEstimator(),
                                               head_pose=AdasHeadPoseEstimator())

        self.add_nodes(self.input, self.network)

    def _process(self):
        ts, frame = self.input.read()

        if frame is None:
            return

        results = self.network.estimate(frame)
        for result in results:
            result.annotate(frame)

        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(15) & 0xFF == 27:
            self.close()

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass


def main():
    pipeline = CascadeFaceDetectionExample(args.input(), multi_threaded=False)
    pipeline.configure(args)
    pipeline.open()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Cascade Face Detection Example", description="Example Pipeline")
    add_logging_parameter(parser)
    input_group = parser.add_argument_group("input provider")
    add_input_step_choices(input_group)

    args = parser.parse_args()

    main()
