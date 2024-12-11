import cv2

from visiongraph import vg


def main():
    image = cv2.imread("media/face.jpg")

    with vg.MediaPipeFaceMeshEstimator() as nn:
        nn: vg.MediaPipeFaceMeshEstimator

        results = nn.process(image)
        results.annotate(image)
        vg.OneEuroFilter
        result = results[0]

        indices = vg.BlazeFaceMesh.LIPS_INDICES

        h, w = image.shape[:2]
        for i in indices:
            lm = result.landmarks[i]
            pos = (round(lm.x * w), round(lm.y * h))
            cv2.circle(image, pos, 5, (255, 255, 255), thickness=1)
            cv2.putText(image, f"{i}", pos, cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

        cv2.imwrite("face-annotated.png", image)
        cv2.imshow("Facemesh", image)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
