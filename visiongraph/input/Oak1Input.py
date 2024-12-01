from typing import Optional

import numpy as np

from visiongraph.input.DepthAIBaseInput import DepthAIBaseInput


class Oak1Input(DepthAIBaseInput):
    """
    A class that represents an input interface for the Oak1 device,
    inheriting from the DepthAIBaseInput class.
    """

    def read(self) -> (int, Optional[np.ndarray]):
        """
        Reads the latest timestamp and RGB frame from the Oak1 device.

        This method overrides the read function in the base class to
        incorporate post-processing of the retrieved data.

        :return: A tuple containing the last timestamp (int)
        """
        super().read()
        return self._post_process(self._last_ts, self._last_rgb_frame)
