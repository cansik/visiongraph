import argparse

import cv2
import numpy as np

from visiongraph import vg


def annotate_classification(results: vg.ResultDict, pose_classifier: vg.FaissKNNClassifier) -> vg.ResultDict:
    """Annotate the image with the closest pose classification."""

    image: np.ndarray = results["image"]
    embeddings: vg.ResultList[vg.LandmarkEmbeddingResult] = results["embeddings"]

    embeddings.annotate(image)

    classifications = pose_classifier.process(embeddings)
    if classifications:
        cls = classifications[0]
        cv2.putText(image, f"{cls.class_name} ({cls.score:.2f})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0))

    return results


def main():
    parser = argparse.ArgumentParser("Pose Classification Example", description="Example Pipeline")
    parser.add_argument("--data", required=True, help="Path to a trained pose-classification dataset.")
    vg.VisionGraph.add_params(parser)
    args = parser.parse_args()

    pose_classifier = vg.FaissKNNClassifier(data_path=args.data)
    pose_classifier.setup()
    if not pose_classifier.labels:
        raise ValueError(f"Pose-classification dataset contains no labels: {args.data}")

    pipeline = (
        vg.create_graph(name="Pose Classification", handle_signals=True)
        .apply(
            image=vg.passthrough(),
            embeddings=vg.sequence(vg.MediaPipePoseEstimator(), vg.LandmarkEmbedder(vg.embed_pose)),
        )
        .then(vg.custom(annotate_classification, pose_classifier), vg.ImagePreview("Pose Classification"))
        .build()
    )
    pipeline.configure(args)

    pipeline.open()
    pipeline.close()


if __name__ == "__main__":
    main()
