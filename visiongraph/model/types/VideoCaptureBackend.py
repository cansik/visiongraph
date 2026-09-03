import cv2

"""
    A constant representing the VideoCapture backend that uses the most suitable one.

    Available backends depend on the constants exposed by the installed OpenCV build:
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
_BACKEND_CONSTANTS = {
    "any": "CAP_ANY",
    "vfw": "CAP_VFW",
    "v4l": "CAP_V4L",
    "v4l2": "CAP_V4L2",
    "firewire": "CAP_FIREWIRE",
    "fireware": "CAP_FIREWARE",
    "ieee1394": "CAP_IEEE1394",
    "dc1394": "CAP_DC1394",
    "cmu1394": "CAP_CMU1394",
    "qt": "CAP_QT",
    "unicap": "CAP_UNICAP",
    "dshow": "CAP_DSHOW",
    "pvapi": "CAP_PVAPI",
    "openni": "CAP_OPENNI",
    "openni_asus": "CAP_OPENNI_ASUS",
    "android": "CAP_ANDROID",
    "xiapi": "CAP_XIAPI",
    "avfoundation": "CAP_AVFOUNDATION",
    "giganetix": "CAP_GIGANETIX",
    "msmf": "CAP_MSMF",
    "winrt": "CAP_WINRT",
    "intelperc": "CAP_INTELPERC",
    "openni2": "CAP_OPENNI2",
    "openni2_asus": "CAP_OPENNI2_ASUS",
    "gphoto2": "CAP_GPHOTO2",
    "gstreamer": "CAP_GSTREAMER",
    "ffmpeg": "CAP_FFMPEG",
    "images": "CAP_IMAGES",
    "aravis": "CAP_ARAVIS",
    "opencv_mjpeg": "CAP_OPENCV_MJPEG",
    "intel_mfx": "CAP_INTEL_MFX",
    "xine": "CAP_XINE",
}


def _available_video_capture_backends(cv2_module: object) -> dict[str, int]:
    """Return video capture backends supported by an OpenCV module."""
    return {
        backend_name: getattr(cv2_module, constant_name)
        for backend_name, constant_name in _BACKEND_CONSTANTS.items()
        if hasattr(cv2_module, constant_name)
    }


VideoCaptureBackend = _available_video_capture_backends(cv2)
