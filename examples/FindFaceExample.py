import argparse
from argparse import ArgumentParser
from typing import Optional

import cv2
import numpy as np

from visiongraph.BaseGraph import BaseGraph
from visiongraph.estimator.spatial.SpatialCascadeEstimator import SpatialCascadeEstimator
from visiongraph.estimator.spatial.face.AdasFaceDetector import AdasFaceDetector
from visiongraph.estimator.spatial.face.landmark.RegressionLandmarkEstimator import RegressionLandmarkEstimator
from visiongraph.estimator.spatial.face.pose.AdasHeadPoseEstimator import AdasHeadPoseEstimator
from visiongraph.estimator.spatial.face.recognition.FaceReidentificationEstimator import FaceReidentificationEstimator
from visiongraph.input import add_input_step_choices
from visiongraph.input.BaseInput import BaseInput
from visiongraph.util.LoggingUtils import add_logging_parameter


class FindFaceExample(BaseGraph):

    def __init__(self, input: BaseInput, target_image: np.ndarray):
        super().__init__()
        self.input = input
        self.network = SpatialCascadeEstimator(AdasFaceDetector.create(),
                                               landmarks=RegressionLandmarkEstimator())
        self.recognition_net = FaceReidentificationEstimator.create()

        self.target_image = target_image
        self.target_embeddings: Optional[np.ndarray] = None

        self.add_nodes(self.input, self.network, self.recognition_net)

    def _init(self):
        super()._init()

        # calculate target embeddings
        results = self.network.process(self.target_image)
        self.target_embeddings = self.recognition_net.process_detection(self.target_image, results[0]).embeddings

    def _process(self):
        ts, frame = self.input.read()

        if frame is None:
            return

        results = self.network.process(frame)

        distances = []

        for result in results:
            embedding = self.recognition_net.process_detection(frame, result)
            distance = embedding.cosine_dist(self.target_embeddings)
            distances.append(distance)

        for i, result in enumerate(results):
            distance = distances[i]
            color = (255, 255, 255)

            if distance < 0.1:
                color = (0, 255, 0)

            result.annotate(frame, color=color, info_text=f"{distance:0.2f}")

        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(15) & 0xFF == 27:
            self.close()

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass


def main():
    target_image = cv2.imread(args.target)

    pipeline = FindFaceExample(args.input(), target_image=target_image)
    pipeline.configure(args)
    pipeline.open()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Find Face Example", description="Example Pipeline")
    add_logging_parameter(parser)
    input_group = parser.add_argument_group("input provider")
    add_input_step_choices(input_group)

    parser.add_argument("--target", required=True, type=str, help="An image of the target face.")

    args = parser.parse_args()
    main()
