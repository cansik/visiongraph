import cv2

"""
    A constant representing the VideoCapture backend that uses the most suitable one.

    Available backends:
    - any: Uses the most suitable backend.
    - vfw, v4l, v4l2: Use VFW, V4L and V4L2 capture drivers respectively.
    - firewire, fireware: Use FireWire or FireWire with Intel QuickStart Technology respectively.
    - ieee1394: Use IEEE 1394 capture driver.
    - dc1394: Use DC1394 capture driver.
    - cmu1394: Use CMU-1394 capture driver.
    - qt: Use Qt capture driver.
    - unicap: Use UNICAP capture driver.
    - dshow, pvapi: Use DirectShow or PVAPI capture drivers respectively.
    - openni, openni_asus: Use OpenNI or OpenNI with Asus 2.0 SDK respectively.
    - android: Use Android camera capture driver.
    - xiapi, avfoundation: Use XIAPI or AVFoundation capture drivers respectively.
    - giganetix: Use Gigabit Ethernet camera capture driver.
    - msmf, winrt, intelperc: Use MSMF, WinRT or Intel Percapture drivers respectively.
    - openni2, openni2_asus: Use OpenNI2 or OpenNI2 with Asus 2.0 SDK respectively.
    - gphoto2, gstreamer, ffmpeg, images: Use GPhoto2, GStreamer, FFmpeg or Images capture drivers respectively.
    - aravis: Use Aravis camera capture driver.
    - opencv_mjpeg: Use Opencv MJPEG-2001 video codec.
    - intel_mfx: Use Intel Media SDK.
    - xine: Use Xine video player capture driver.
    """
# https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html#gga023786be1ee68a9105bf2e48c700294dacf10e9692c4166f74de62b7d00c377d0
VideoCaptureBackend = {
    "any": cv2.CAP_ANY,
    "vfw": cv2.CAP_VFW,
    "v4l": cv2.CAP_V4L,
    "v4l2": cv2.CAP_V4L2,
    "firewire": cv2.CAP_FIREWIRE,
    "fireware": cv2.CAP_FIREWARE,
    "ieee1394": cv2.CAP_IEEE1394,
    "dc1394": cv2.CAP_DC1394,
    "cmu1394": cv2.CAP_CMU1394,
    "qt": cv2.CAP_QT,
    "unicap": cv2.CAP_UNICAP,
    "dshow": cv2.CAP_DSHOW,
    "pvapi": cv2.CAP_PVAPI,
    "openni": cv2.CAP_OPENNI,
    "openni_asus": cv2.CAP_OPENNI_ASUS,
    "android": cv2.CAP_ANDROID,
    "xiapi": cv2.CAP_XIAPI,
    "avfoundation": cv2.CAP_AVFOUNDATION,
    "giganetix": cv2.CAP_GIGANETIX,
    "msmf": cv2.CAP_MSMF,
    "winrt": cv2.CAP_WINRT,
    "intelperc": cv2.CAP_INTELPERC,
    "openni2": cv2.CAP_OPENNI2,
    "openni2_asus": cv2.CAP_OPENNI2_ASUS,
    "gphoto2": cv2.CAP_GPHOTO2,
    "gstreamer": cv2.CAP_GSTREAMER,
    "ffmpeg": cv2.CAP_FFMPEG,
    "images": cv2.CAP_IMAGES,
    "aravis": cv2.CAP_ARAVIS,
    "opencv_mjpeg": cv2.CAP_OPENCV_MJPEG,
    "intel_mfx": cv2.CAP_INTEL_MFX,
    "xine": cv2.CAP_XINE
}
