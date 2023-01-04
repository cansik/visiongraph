import cv2

from visiongraph import FPSTracer

fps_tracer = FPSTracer()
camera = cv2.VideoCapture(0)

codec = 0x47504A4D  # MJPG
camera.set(cv2.CAP_PROP_FPS, 60)
camera.set(cv2.CAP_PROP_CODEC_PIXEL_FORMAT, codec)
camera.set(cv2.CAP_PROP_FOURCC, codec)
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m', 'j', 'p', 'g'))
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while 1:
    retval, im = camera.read(0)

    fps_tracer.update()

    cv2.putText(im, "FPS: %.0f" % fps_tracer.smooth_fps,
                (7, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("image", im)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

camera.release()
cv2.destroyAllWindows()
