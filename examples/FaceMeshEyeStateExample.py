import argparse

import cv2
import numpy as np

from visiongraph import vg


def main():
    parser = argparse.ArgumentParser(
        "Face Mesh Eye State Example", description="Classify each detected eye as open or closed."
    )
    vg.VisionGraph.add_params(parser)
    args = parser.parse_args()

    face_mesh = vg.MediaPipeFaceMeshEstimator()
    face_mesh.setup()

    eye_classifier = vg.EyeOpenClosedEstimator()
    eye_classifier.setup()

    def annotate_eye_states(image: np.ndarray) -> np.ndarray:
        faces = face_mesh.process(image)
        if not faces:
            return image

        face = faces[0]

        for name, indices in [("left", face.LEFT_EYE_BOX_INDICES), ("right", face.RIGHT_EYE_BOX_INDICES)]:
            eye_box = vg.bbox_from_landmarks(face.landmarks[indices]).scale_centered(0.3, 0.3)
            roi, _, _ = vg.roi_safe(image, eye_box, rectified=True)

            result = eye_classifier.process(roi)
            vg.draw_bbox(image, eye_box)
            vg.draw_text_normalized(
                image,
                f"{name}: {result.class_name}",
                eye_box.top_left,
                font=cv2.FONT_HERSHEY_PLAIN,
            )

        return image

    graph = (
        vg.create_graph(name="Face Mesh Eye State", input_node=args.input(), handle_signals=True)
        .then(vg.custom(annotate_eye_states), vg.ImagePreview())
        .build()
    )
    graph.configure(args)
    graph.open()


if __name__ == "__main__":
    main()
