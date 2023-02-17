import argparse
from typing import Sequence

import cv2
import numpy as np
import vector

import visiongraph as vg


class Face3D(vg.BaseResult):
    def __init__(self, face: vg.BlazeFaceMesh, distance: float):
        self.face = face
        self.distance = distance

    def annotate(self, image: np.ndarray, **kwargs):
        h, w = image.shape[:2]

        def mark_point(point: vector.Vector2D):
            x = int(point.x * w)
            y = int(point.y * h)

            cv2.circle(image, (x, y), 5, (0, 255, 255), 1)

        # self.face.annotate(image, **kwargs)
        mark_point(self.face.left_iris.to_xy())
        mark_point(self.face.right_iris.to_xy())

        br = self.face.bounding_box.bottom_right
        x = int(br.x * w) + 5
        y = int(br.y * h)

        cv2.putText(image, f"{self.distance:.2f}m", (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255))


class IrisDistanceApp:
    def __init__(self, args):
        self._input_width: int = 640
        self._input_height: int = 480

        self.average_iris_radius = 11.7

        # this should be measured by calibration process
        self.normalized_focale_x = 1.40625

        # define graph
        self.graph = (
            vg.create_graph(name="FaceMesh Example", input_node=args.input(), handle_signals=True)

            # run detection and pass image through for annotation
            .then(vg.custom(self.read_input))
            .apply(
                image=vg.passthrough(),
                face_mesh=vg.sequence(vg.MediaPipeFaceMeshEstimator(refine_landmarks=True),
                                      vg.custom(self.process_faces)),
            )

            # annotate result
            .then(
                vg.ResultAnnotator(),
                vg.ImagePreview("Preview")
            )
            .build()
        )
        self.graph.configure(args)

    def run(self):
        # start graph
        self.graph.open()

    def read_input(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        self._input_width = w
        self._input_height = h
        return image

    def process_faces(self, faces: vg.ResultList[vg.BlazeFaceMesh]) -> vg.ResultList[Face3D]:
        faces_3d = vg.ResultList([self.process_face(face) for face in faces])
        return faces_3d

    def process_face(self, face: vg.BlazeFaceMesh) -> Face3D:
        left_distance = self.measure_iris_distance_in_m(face, face.LEFT_IRIS_INDICES)
        right_distance = self.measure_iris_distance_in_m(face, face.RIGHT_IRIS_INDICES)

        avg_distance = np.mean([left_distance, right_distance])

        return Face3D(face, avg_distance)

    def measure_iris_distance_in_m(self, face: vg.BlazeFaceMesh, iris_indices: Sequence[int]) -> float:
        x_values = np.array([face.landmarks[i].x for i in iris_indices]) * self._input_width
        min_x = np.min(x_values)
        max_x = np.max(x_values)

        d = abs(max_x - min_x)
        fx = min(self._input_width, self._input_height) * self.normalized_focale_x
        d_z = (fx * (self.average_iris_radius / d)) / 1000.0

        return d_z


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser("Iris Distance Example", description="Detect distance of faces.")
    vg.VisionGraph.add_params(parser)
    args = parser.parse_args()

    app = IrisDistanceApp(args)
    app.run()


if __name__ == "__main__":
    main()
