from dataclasses import dataclass
from enum import Enum
from typing import Optional


class NMSBatchMode(Enum):
    Auto = 1 << 0
    Disabled = 1 << 1
    Enabled = 1 << 2


@dataclass
class NMSOptions:
    """
    :param enabled: Flag indicating whether to apply non-maximum suppression (default is True).
    :param score_threshold: Threshold value for confidence score (default is 0.3).
    :param nms_threshold: Threshold value for non-maximum suppression (IoU overlap) (default is 0.3).
    :param eta: Optional parameter for non-maximum suppression.
    :param top_k: Optional parameter for non-maximum suppression.
    """

    enabled: bool = True
    score_threshold: float = 0.3
    nms_threshold: float = 0.3
    eta: Optional[float] = None
    top_k: Optional[int] = None
    batch_mode: NMSBatchMode = NMSBatchMode.Auto
