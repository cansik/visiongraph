import argparse
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

from visiongraph import vg
from visiongraph.model.RollingBufferNumpy import RollingBufferNumpy


@dataclass
class VVADOptions:
    """
    Configuration options for VivaVAD.

    :param sequence_length: The length of the input sequence for VAD.
    :param landmark_indices: Indices of facial landmarks used for processing.
    :param min_score: Minimum score to classify as speaking.
    """
    sequence_length: int = 10
    landmark_indices: Sequence[int] = vg.BlazeFaceMesh.FEATURES_148
    min_score: float = 0.75

    landmark_indices_numpy: Optional[np.ndarray] = None

    def __post_init__(self):
        # pre-calculate this indices array
        self.landmark_indices_numpy = np.array(list(self.landmark_indices), dtype=np.uint32)


@dataclass
class TrackedFace(vg.BaseResult, vg.Trackable):
    """
    Represents a tracked face with VAD results.

    :param vvad_options: VAD configuration options.
    :param face_mesh: The detected face mesh data.
    :param feature_buffer: Buffer to store processed facial features.
    :param vvad_result: The VivaVAD classification result.
    """
    vvad_options: VVADOptions
    face_mesh: vg.BlazeFaceMesh
    feature_buffer: Optional[RollingBufferNumpy] = None
    vvad_result: Optional[vg.VivaVADResult] = None
    filtered_speaking_score: Optional[vg.OneEuroFilter] = None

    @property
    def tracking_id(self) -> int:
        """
        Returns the tracking ID associated with the face mesh.
        """
        return self.face_mesh.tracking_id

    def update_track(self, track: "TrackedFace"):
        """
        Updates the tracked face data with new observations.

        :param track: The tracked face data to update from.
        """
        # lazy initialize buffer
        if track.feature_buffer is None:
            track.feature_buffer = RollingBufferNumpy(self.vvad_options.sequence_length,
                                                      len(self.vvad_options.landmark_indices_numpy) * 3,
                                                      dtype=np.float32)

        if track.filtered_speaking_score is None:
            track.filtered_speaking_score = vg.OneEuroFilter(0.0, min_cutoff=2.0, beta=0.0, d_cutoff=0.1)

        if track.vvad_result is not None:
            track.filtered_speaking_score(track.vvad_result.speaking_score)

        # update values
        track.face_mesh = self.face_mesh
        track.feature_buffer.add(self._pre_process_face(self.face_mesh))

    def _pre_process_face(self, face: vg.BlazeFaceMesh) -> np.ndarray:
        """
        Preprocesses a face mesh for VAD prediction.

        :param face: The face mesh to process.

        :return: A flattened NumPy array of preprocessed landmarks.
        """
        if face.transformation_matrix is None:
            raise Exception("Output facial transformation matrices must be enabled in MediaPipeFaceMeshEstimator.")

        # normalize (use 1-batching, since it matches the default per-processing)
        x = np.expand_dims(face.normalize_landmarks(), 0)

        # extract indices
        x = x[:, self.vvad_options.landmark_indices_numpy]

        # flatten landmarks
        x = x.reshape(x.shape[0], -1)
        return x[0].astype(np.float32)

    def annotate(self, image: np.ndarray, **kwargs) -> None:
        """
        Annotates the image with tracking and VAD information.

        :param image: The image to annotate.
        :param kwargs: Additional arguments for annotation.
        """
        is_speaking = False
        bbox = self.face_mesh.bounding_box

        if self.vvad_result is not None:
            score = self.filtered_speaking_score.x_prev
            raw = self.vvad_result.speaking_score

            is_speaking = score > self.vvad_options.min_score
            vg.draw_text_normalized(image, f"{score * 100:.0f}%",
                                    bbox.top_right, font=cv2.FONT_HERSHEY_PLAIN)

        box_color = (0, 255, 0) if is_speaking else (0, 0, 255)
        vg.draw_bbox(image, bbox, color=box_color)

        # self.face_mesh.annotate(image, landmark_colors=landmark_colors)


class VVADApplication:
    """
    Application to detect face meshes and perform voice activity detection (VAD).
    """

    def __init__(self, args: argparse.Namespace):
        """
        Initializes the VVADApplication with user-provided arguments.

        :param args: Command-line arguments.
        """
        self.args = args
        self.vvad_options = VVADOptions()
        self.vvad = vg.VivaVAD.create()

        self.graph = (
            vg.create_graph(name="FaceMesh VVAD Example", input_node=args.input(), handle_signals=True)

            # run detection and pass image through for annotation
            .apply(
                image=vg.passthrough(),

                # detect face meshes, run vvad detection and filter results
                face_meshes=vg.sequence(
                    # detect face meshes and track them
                    vg.MediaPipeFaceMeshEstimator(max_num_faces=2, output_facial_transformation_matrixes=True),
                    vg.FlateTracker(),

                    # convert results into VVADFace and update filters (using tracking storage)
                    vg.custom(self._as_tracked_faces),
                    vg.SimpleTrackingStorage(),

                    # run vvad prediction per face
                    vg.custom(self._predict_vad)
                ),
            )

            # annotate result
            .then(
                vg.ResultAnnotator(),
                vg.ImagePreview("Viva VVAD")
            )
            .build()
        )
        self.graph.unlinked_nodes.append(self.vvad)

    def _predict_vad(self, faces: vg.ResultList[TrackedFace]) -> vg.ResultList[TrackedFace]:
        """
        Performs VAD prediction on tracked faces.

        :param faces: List of tracked faces.

        :return: The updated list of faces with VAD results.
        """
        valid_faces = [face for face in faces if face.feature_buffer is not None]
        if len(valid_faces) == 0:
            return faces

        results = self.vvad.process([face.feature_buffer.get() for face in valid_faces])

        # update faces with results
        for face, result in zip(valid_faces, results):
            face.vvad_result = result

        return faces

    def _as_tracked_faces(self, faces: vg.ResultList[vg.BlazeFaceMesh]) -> vg.ResultList[TrackedFace]:
        """
        Converts BlazeFaceMesh results into tracked faces.

        :param faces: List of detected face meshes.

        :return: List of tracked face objects.
        """
        return vg.ResultList([TrackedFace(self.vvad_options, face_mesh) for face_mesh in faces])

    def run(self):
        """
        Configures and starts the VVAD application.
        """
        self.graph.configure(self.args)
        self.graph.open()


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser("FaceMesh Example", description="Detect face meshes on images.")
    vg.VisionGraph.add_params(parser)
    args = parser.parse_args()

    app = VVADApplication(args)
    app.run()


if __name__ == "__main__":
    main()
