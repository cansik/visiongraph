import argparse

import cv2
import numpy as np
import vector

from visiongraph import vg


class IrisDistanceApp:
    def __init__(self, args):
        self.intrinsics = vg.CameraIntrinsics.load(args.intrinsics)
        self.iris_distance_calculator = vg.IrisDistanceCalculator(-1, -1, self.intrinsics)

        # define graph
        self.graph = (
            vg.create_graph(name="Iris Distance Calculator Example", input_node=args.input(), handle_signals=True)
            # run detection and pass image through for annotation
            .then(vg.custom(self.read_input))
            .apply(
                image=vg.passthrough(),
                iris=vg.sequence(vg.MediaPipeFaceMeshEstimator(refine_landmarks=True), self.iris_distance_calculator),
            )
            # annotate result
            .then(vg.ResultAnnotator(), vg.custom(self._annotate_distance), vg.ImagePreview("Preview"))
            .build()
        )
        self.graph.configure(args)

    def run(self):
        # start graph
        self.graph.open()

    def read_input(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        self.iris_distance_calculator.input_width = w
        self.iris_distance_calculator.input_height = h
        return image

    def _annotate_distance(self, results: vg.ResultDict):
        image: np.ndarray = results["image"]
        h, w = image.shape[:2]
        iris_results: vg.ResultList[vg.IrisDistanceResult] = results["iris"]

        if len(iris_results) == 0:
            return

        iris_result = iris_results[0]

        # calculate projected point
        head_center = iris_result.head_center()
        head_center_location = vector.obj(x=head_center.x * w, y=head_center.y * h)

        point = vg.project_pixel_to_point(head_center_location, iris_result.average_iris_distance(), self.intrinsics)

        cv2.putText(
            image,
            f"{iris_result.average_iris_distance():.2f}m (x={point.x:.2f}, y={point.y:.2f}, z={point.z:.2f})",
            (20, 50),
            cv2.FONT_HERSHEY_DUPLEX,
            0.6,
            (255, 0, 255),
        )


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser("Iris Distance Example", description="Detect distance of faces.")
    parser.add_argument("--intrinsics", type=str, help="Camera intrinsics parameter.")
    vg.VisionGraph.add_params(parser)
    args = parser.parse_args()

    app = IrisDistanceApp(args)
    app.run()


if __name__ == "__main__":
    main()
