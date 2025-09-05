import argparse

import cv2
import numpy as np

from visiongraph import vg

pose_data_path = "media/pose_training.npz"
pose_classifier = vg.FaissKNNClassifier(data_path=pose_data_path)


def classify(results: vg.ResultDict):
    image: np.ndarray = results["image"]
    embeddings: vg.ResultList[vg.LandmarkEmbeddingResult] = results["embeddings"]

    embeddings.annotate(image)

    classifications = pose_classifier.process(embeddings)
    if len(classifications) > 0:
        cls = classifications[0]
        cv2.putText(image, f"{cls.class_name} ({cls.score:.2f})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0))

    cv2.imshow("Pose", image)
    key = cv2.waitKey(5) & 0xFF

    if key == 27:
        quit(0)

    if chr(key).lower() == "s":
        print("saving samples")
        pose_classifier.save_data(pose_data_path)

    if chr(key).lower() == "l":
        print("loading samples")
        pose_classifier.load_data(pose_data_path)

    if chr(key).isnumeric():
        number = int(chr(key))

        if len(embeddings) > 0:
            print(f"sample class: {number}")
            embedding = embeddings[0]
            pose_classifier.add_sample(embedding, number)


def main():
    parser = argparse.ArgumentParser("Pose Classification Example", description="Example Pipeline")
    vg.VisionGraph.add_params(parser)
    args = parser.parse_args()

    pose_classifier.setup()

    if len(pose_classifier.labels) == 0:
        pose_classifier.labels.append("standing")
        pose_classifier.labels.append("sitting")

    pipeline = (
        vg.create_graph(name="Pose Classification", handle_signals=True)
        .apply(
            image=vg.passthrough(),
            embeddings=vg.sequence(vg.MediaPipePoseEstimator(), vg.LandmarkEmbedder(vg.embed_pose)),
        )
        .then(vg.custom(classify))
        .build()
    )
    pipeline.configure(args)

    pipeline.open()
    pipeline.close()


if __name__ == "__main__":
    main()
