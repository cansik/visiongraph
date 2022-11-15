import cv2
import vector

import visiongraph as vg
from visiongraph.util import ImageUtils

iw, ih = 224, 224
image = cv2.imread("media/hand.jpg")
h, w = image.shape[:2]

input, padding_box = ImageUtils.resize_and_pad(image, (iw, ih))

result = vg.LandmarkDetectionResult(0, "face", 1.0, vector.array(
    {
        "x": [114 / iw],
        "y": [43 / ih],
        "z": [0.0],
        "t": [1.0],
    }
), vg.BoundingBox2D(105 / iw, 31 / ih, 18 / iw, 20 / ih))
result.tracking_id = 2

output = input.copy()
result.annotate(input, show_bounding_box=True)
cv2.imshow("Input", input)

result.map_coordinates((iw, ih), (w, h), src_roi=padding_box)

back_image = image.copy()
result.annotate(image, show_bounding_box=True)
cv2.imshow("Result", image)

# map back
input_box = vg.BoundingBox2D(0, 0, w, h)
result.map_coordinates((w, h), (iw, ih), dest_roi=padding_box)

result.annotate(output, show_bounding_box=True)
cv2.imshow("Back Map", output)

cv2.waitKey(0)
cv2.destroyAllWindows()
