import numpy as np
from mediapipe.tasks.python import vision

from visiongraph.data.Asset import Asset
from visiongraph.data.AssetMetadata import AssetMetadata
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.estimator.spatial.mediapipe.MediaPipeTaskRunner import MediaPipeTaskRunner
from visiongraph.estimator.spatial.face.landmark.FaceLandmarkEstimator import FaceLandmarkEstimator
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.face.BlazeFaceMesh import BlazeFaceMesh
from visiongraph.result.spatial.face.BlendShape import BlendShape
from visiongraph.util.VectorUtils import list_of_vector4D


_MODEL_METADATA = AssetMetadata.from_values(
    source_name="MediaPipe face landmarker tasks",
    source_url="https://github.com/google-ai-edge/mediapipe",
    license_name="Apache License 2.0",
    license_url="https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE",
)
_DEFAULT_TASK = RepositoryAsset("face_landmarker.task", metadata=_MODEL_METADATA)


class MediaPipeFaceMeshEstimator(FaceLandmarkEstimator[BlazeFaceMesh]):
    """Detect facial landmarks, blend shapes, and transformation matrices with MediaPipe Face Landmarker."""

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_faces: int = 1,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        output_face_blendshapes: bool = False,
        output_facial_transformation_matrixes: bool = False,
        task: Asset = _DEFAULT_TASK,
    ) -> None:
        """
        Initialize the face mesh estimator.

        :param static_image_mode: Whether to detect faces independently in each image.
        :param max_num_faces: The maximum number of faces to detect.
        :param min_face_detection_confidence: The minimum detection confidence score.
        :param min_face_presence_confidence: The minimum confidence for face presence detection.
        :param min_tracking_confidence: The minimum confidence for tracking landmarks.
        :param output_face_blendshapes: Whether to output face blend shapes.
        :param output_facial_transformation_matrixes: Whether to output facial transformation matrices.
        :param task: MediaPipe task asset to use for face mesh estimation.
        """
        super().__init__(min_face_detection_confidence)
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.min_face_presence_confidence = min_face_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.output_face_blendshapes = output_face_blendshapes
        self.output_facial_transformation_matrixes = output_facial_transformation_matrixes
        self.task = task
        self._task = MediaPipeTaskRunner(
            vision.FaceLandmarkerOptions,
            vision.FaceLandmarker.create_from_options,
            vision.FaceLandmarker.detect,
            vision.FaceLandmarker.detect_for_video,
        )

    def _options(self) -> dict[str, vision.RunningMode | int | float | bool]:
        return dict(
            running_mode=vision.RunningMode.IMAGE if self.static_image_mode else vision.RunningMode.VIDEO,
            num_faces=self.max_num_faces,
            min_face_detection_confidence=self.min_score,
            min_face_presence_confidence=self.min_face_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            output_face_blendshapes=self.output_face_blendshapes,
            output_facial_transformation_matrixes=self.output_facial_transformation_matrixes,
        )

    def setup(self) -> None:
        """Set up the face landmarker."""
        self._task.setup(self.task, **self._options())

    def process(self, image: np.ndarray, timestamp_ms: int | None = None) -> ResultList[BlazeFaceMesh]:
        """
        Detect faces and estimate landmarks.

        Option changes take effect on the next frame and reset tracking.

        :param image: The input image as a NumPy array in BGR format.
        :param timestamp_ms: A strictly increasing video timestamp in milliseconds, or None for automatic timestamps.
        :return: The landmarks and requested face attributes for each detection.
        """
        results = self._task.process(image, self.task, timestamp_ms, **self._options())
        faces: ResultList[BlazeFaceMesh] = ResultList()
        for index, face_landmarks in enumerate(results.face_landmarks):
            landmarks = list_of_vector4D([(point.x, point.y, point.z, 1.0) for point in face_landmarks])
            face_mesh = BlazeFaceMesh(1.0, landmarks)

            if results.face_blendshapes:
                blend_shapes = results.face_blendshapes[index]
                face_mesh.blend_shapes = [BlendShape(b.index, b.category_name, b.score) for b in blend_shapes]

            if results.facial_transformation_matrixes:
                face_mesh.transformation_matrix = results.facial_transformation_matrixes[index]

            faces.append(face_mesh)
        return faces

    def release(self) -> None:
        """Release the task's resources."""
        self._task.release()
