import argparse
from argparse import ArgumentParser

import cv2

from visiongraph.BaseGraph import BaseGraph
from visiongraph.estimator.spatial.SpatialCascadeEstimator import SpatialCascadeEstimator
from visiongraph.estimator.spatial.face.AdasFaceDetector import AdasFaceDetector
from visiongraph.estimator.spatial.face.landmark.RegressionLandmarkEstimator import RegressionLandmarkEstimator
from visiongraph.estimator.spatial.face.pose.AdasHeadPoseEstimator import AdasHeadPoseEstimator
from visiongraph.estimator.spatial.face.recognition.FaceReidentificationEstimator import FaceReidentificationEstimator
from visiongraph.input import add_input_step_choices
from visiongraph.input.BaseInput import BaseInput
from visiongraph.util.LoggingUtils import add_logging_parameter


class FaceRecognitionExample(BaseGraph):

    def __init__(self, input: BaseInput):
        super().__init__()
        self.input = input
        self.network = SpatialCascadeEstimator(AdasFaceDetector.create(),
                                               landmarks=RegressionLandmarkEstimator())
        self.recognition_net = FaceReidentificationEstimator.create()

        self.add_nodes(self.input, self.network, self.recognition_net)

    def _process(self):
        ts, frame = self.input.read()

        if frame is None:
            return

        results = self.network.process(frame)

        for result in results:
            embedding = self.recognition_net.process_detection(frame, result)
            print(embedding)

        for result in results:
            result.annotate(frame)

        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(15) & 0xFF == 27:
            self.close()

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass


def main():
    pipeline = FaceRecognitionExample(args.input())
    pipeline.configure(args)
    pipeline.open()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Cascade Face Detection Example", description="Example Pipeline")
    add_logging_parameter(parser)
    input_group = parser.add_argument_group("input provider")
    add_input_step_choices(input_group)

    args = parser.parse_args()

    main()
