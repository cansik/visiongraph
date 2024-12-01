import logging
from typing import Union, Optional, Tuple

import cv2
import numpy as np
from vidgear.gears import CamGear

from visiongraph.input.VideoCaptureInput import VideoCaptureInput


class CamGearInput(VideoCaptureInput):
    """
    A class that handles video capture input from a CamGear source.
    This class extends the VideoCaptureInput to include specific
    setup and management for CamGear inputs.
    """

    def __init__(self, channel: Union[str, int] = 0, input_skip: int = -1,
                 loop: bool = True, fps_lock: bool = False, stream_mode: bool = False):
        """
        Initializes the CamGearInput with the given parameters.

        :param channel: The input channel for video capture (default is 0).
        :param input_skip: The milliseconds to skip in the video stream (default is -1).
        :param loop: Indicates whether to loop the video (default is True).
        :param fps_lock: Indicates whether to lock to a specific frames per second (default is False).
        :param stream_mode: Indicates whether the capture is in stream mode (default is False).
        """
        super().__init__(channel, input_skip, loop, fps_lock)
        self.stream_mode: bool = stream_mode
        self.input_options = {}

        self._cap: Optional[CamGear] = None

    def _setup_cap(self):
        """
        Sets up the video capture device with the specified options.
        Configures resolution, frames per second, and input skipping.
        Starts the CamGear capture.
        """
        if str(self.channel).isnumeric():
            # probably a webcam input
            self.input_options.update({
                "CAP_PROP_FRAME_WIDTH": self.width,
                "CAP_PROP_FRAME_HEIGHT": self.height,
                "CAP_PROP_FPS": self.fps,
            })

        if self.input_skip >= 0:
            self.input_options.update({
                "CAP_PROP_POS_MSEC": self.input_skip
            })

        self._cap = CamGear(source=self.channel, **self.input_options)
        self._cap.start()

        if not self._is_cap_open():
            logging.warning("Could not open CamGear, please check if channel is correct.")

        self.fps = self._cap.framerate

    def _release_cap(self):
        """
        Releases the captured video device.
        Stops the CamGear capture.
        """
        self._cap.stop()

    def _is_cap_open(self) -> bool:
        """
        Checks if the video capture device is open.

        :return: True if the capture device is open, False otherwise.
        """
        return self._cap is not None and self._cap.stream.isOpened()

    def _read_next_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Reads the next frame from the video capture.

        :return: A tuple containing a boolean indicating
        """
        frame = self._cap.read()
        return frame is not None, frame

    def _skip_to_frame(self, frame_position: int):
        """
        Skips to a specific frame position in the video stream.

        :param frame_position: The position of the frame to skip to.
        """
        self._cap.stream.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
