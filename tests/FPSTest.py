import time

import cv2

import visiongraph as vg


def main():
    fps_tracer = vg.FPSTracer(0.1)
    running = True

    pose = cv2.imread("media/pose.png")

    slowdown = 1.2

    while running:
        image = pose.copy()

        if slowdown > 0:
            time.sleep(slowdown)

        cv2.putText(image, "FPS: %.0f" % fps_tracer.smooth_fps,
                    (7, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("FPS Test", image)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            running = False
        if key == ord("1"):
            slowdown += 0.05
            print(slowdown)
        if key == ord("2"):
            slowdown -= 0.05
            print(slowdown)

        fps_tracer.update()


if __name__ == "__main__":
    main()
