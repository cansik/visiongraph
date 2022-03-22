import argparse
import os.path
from argparse import ArgumentParser
from typing import Optional, List

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from visiongraph.BaseGraph import BaseGraph
from visiongraph.estimator.spatial.SpatialCascadeEstimator import SpatialCascadeEstimator
from visiongraph.estimator.spatial.face.AdasFaceDetector import AdasFaceDetector
from visiongraph.estimator.spatial.face.landmark.RegressionLandmarkEstimator import RegressionLandmarkEstimator
from visiongraph.estimator.spatial.face.recognition.FaceReidentificationEstimator import FaceReidentificationEstimator
from visiongraph.input import add_input_step_choices
from visiongraph.input.BaseInput import BaseInput
from visiongraph.util.LoggingUtils import add_logging_parameter


class Target:
    name: str
    image: np.ndarray
    embeddings: Optional[np.ndarray]


class FindFaceExample(BaseGraph):

    def __init__(self, input: BaseInput):
        super().__init__()
        self.input = input
        self.network = SpatialCascadeEstimator(AdasFaceDetector.create(),
                                               landmarks=RegressionLandmarkEstimator())
        self.recognition_net = FaceReidentificationEstimator.create()
        self.add_nodes(self.input, self.network, self.recognition_net)

        self.targets: List[Target] = []

    def _init(self):
        super()._init()

        # calculate target embeddings
        for target in self.targets:
            results = self.network.process(target.image)
            target.embeddings = self.recognition_net.process_detection(target.image, results[0]).embeddings

    def _process(self):
        ts, frame = self.input.read()

        if frame is None:
            return

        results = self.network.process(frame)

        # estimate face embeddings for results
        result_embeddings = []
        for tr, result in enumerate(results):
            embedding = self.recognition_net.process_detection(frame, result)
            result_embeddings.append(embedding.embeddings)

        # calculate cost matrix
        target_embeddings = np.array([t.embeddings for t in self.targets])
        result_embeddings = np.array(result_embeddings)
        costs = cdist(result_embeddings, target_embeddings, "cosine") * 0.5

        # solve linear assignment
        row_ind, col_ind = linear_sum_assignment(costs)
        matching_error = costs[row_ind, col_ind].sum() / costs.shape[0]
        lookup_table = dict(zip(row_ind.tolist(), col_ind.tolist()))

        # display results
        for i, result in enumerate(results):
            color = (255, 255, 255)
            info_text: Optional[str] = None

            if i in lookup_table:
                target_index = lookup_table[i]
                target = self.targets[target_index]

                distance = costs[i, target_index]
                color = (0, 255, 0)
                info_text = f"{target.name[:10]} ({distance:0.2f})"

            result.annotate(frame, color=color, info_text=info_text)

        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(15) & 0xFF == 27:
            self.close()

    def configure(self, args: argparse.Namespace):
        super().configure(args)

        for file in args.targets:
            target = Target()
            target.name = os.path.splitext(os.path.basename(file))[0]
            target.image = cv2.imread(file)
            self.targets.append(target)

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass


def main():
    pipeline = FindFaceExample(args.input())
    pipeline.configure(args)
    pipeline.open()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Find Face Example", description="Example Pipeline")
    add_logging_parameter(parser)
    input_group = parser.add_argument_group("input provider")
    add_input_step_choices(input_group)

    parser.add_argument("--targets", required=True, type=str, nargs="+", help="Image paths of the faces to find.")

    args = parser.parse_args()
    main()
