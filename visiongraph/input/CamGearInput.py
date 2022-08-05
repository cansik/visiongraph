import logging
from typing import Union, Optional, Tuple

import cv2
import numpy as np
from vidgear.gears import CamGear

from visiongraph.input.VideoCaptureInput import VideoCaptureInput


class CamGearInput(VideoCaptureInput):
    def __init__(self, channel: Union[str, int] = 0, input_skip: int = -1,
                 loop: bool = True, fps_lock: bool = False, stream_mode: bool = False):
        super().__init__(channel, input_skip, loop, fps_lock)

        self.stream_mode: bool = stream_mode
        self.input_options = {}

        self._cap: Optional[CamGear] = None

    def _setup_cap(self):
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
        self._cap.stop()

    def _is_cap_open(self) -> bool:
        return self._cap is not None and self._cap.stream.isOpened()

    def _read_next_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        frame = self._cap.read()
        return frame is not None, frame

    def _skip_to_frame(self, frame_position: int):
        self._cap.stream.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
