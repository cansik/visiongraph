import argparse
from typing import List, Optional

import cv2
import numpy as np

import visiongraph as vg

FACE_KEY = "face"

pipeline: Optional[vg.VisionGraph] = None

output_size = vg.Size2D(400, 400)
blank_image = np.zeros((output_size.height, output_size.width, 3), dtype=np.uint8)

normalized_keypoints = np.array([
    [0.621875, 0.489453125],  # right eye
    [0.35703125, 0.489453125],  # left eye
    [0.489453125, 0.6603515625],  # tip of nose
    [0.4103515625, 0.568359375],  # right lip corner
    [0.4103515625, 0.7603515625],  # left lip corner
], dtype=np.float32)


def extract_face_texture(result: vg.ResultDict, recognizer: vg.FaceReidentificationEstimator) -> vg.ResultDict:
    image: np.ndarray = result["image"]
    faces: List[vg.FaceLandmarkResult] = result["faces"]

    if len(faces) == 0:
        result[FACE_KEY] = blank_image
        return result

    roi, detection = vg.extract_object_detection_roi(image, faces[0])
    aligned_face, landmark_overlap = recognizer._align_face(roi, detection, normalized_keypoints)
    aligned_face = cv2.resize(aligned_face, (output_size.width, output_size.height))
    result[FACE_KEY] = aligned_face
    return result


def main():
    # hack to use face alignment from face recognition
    recognizer = vg.FaceReidentificationEstimator.create()

    global pipeline
    pipeline = (
        vg.create_graph(name="Face Texture Detection", handle_signals=True)
        .apply(faces=vg.sequence(vg.MediaPipeFaceDetector(), vg.CentroidTracker()),
               image=vg.passthrough())
        .then(vg.custom(extract_face_texture, recognizer), vg.ImagePreview("face", image_key=FACE_KEY))
        .apply(share=vg.sequence(vg.extract(FACE_KEY), vg.FrameBufferSharingServer.create("face-texture")),
               preview=vg.sequence(vg.ResultAnnotator(), vg.ImagePreview()))
        .build()
    )
    pipeline.configure(args)

    pipeline.open()
    pipeline.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Frame Buffer Sharing Example", description="Example Pipeline")
    vg.VisionGraph.add_params(parser)
    vg.CentroidTracker.add_params(parser)
    args = parser.parse_args()

    main()
