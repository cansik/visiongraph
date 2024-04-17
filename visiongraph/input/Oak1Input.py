from typing import Optional

import numpy as np

from visiongraph.input.DepthAIBaseInput import DepthAIBaseInput


class Oak1Input(DepthAIBaseInput):
    def read(self) -> (int, Optional[np.ndarray]):
        super().read()
        return self._post_process(self._last_ts, self._last_rgb_frame)
