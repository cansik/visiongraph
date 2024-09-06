import cv2
import numpy as np

from visiongraph import vg

if __name__ == "__main__":
    face_mesh = vg.MediaPipeFaceMeshEstimator()
    face_mesh.setup()

    eye_classifier = vg.EyeOpenClosedEstimator()
    eye_classifier.setup()


    def check_eye_closed(image: np.ndarray):
        faces = face_mesh.process(image)
        if len(faces) == 0:
            return
        face = faces[0]

        # extract roi
        for name, indices in [("left", face.LEFT_EYE_BOX_INDICES), ("right", face.RIGHT_EYE_BOX_INDICES)]:
            left_box = vg.bbox_from_landmarks(face.landmarks[indices]).scale_centered(0.3, 0.3)
            roi, xs, ys = vg.roi_safe(image, left_box, rectified=True)

            result = eye_classifier.process(roi)
            print(f"{name}: {result.class_name}")

            cv2.imshow(f"ROI {name}", roi)


    graph = (
        vg.create_graph(name="VisionGraph", input_node=vg.VideoCaptureInput(), handle_signals=True)
        .then(vg.custom(check_eye_closed))
        .then(vg.ImagePreview())
    ).open()
