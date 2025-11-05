from dataclasses import dataclass
from enum import Enum
from typing import Optional


class NMSBatchMode(Enum):
    """
    Enumeration for batch processing modes in Non-Maximum Suppression (NMS).

    - Auto: Automatically determine whether to apply batch mode.
    - Disabled: Apply NMS independently for each input.
    - Enabled: Apply NMS across all inputs in a batch.
    """
    Auto = 1 << 0
    Disabled = 1 << 1
    Enabled = 1 << 2


@dataclass
class NMSOptions:
    """
    Configuration options for Non-Maximum Suppression (NMS) used in object detection.

    :param enabled: Flag indicating whether to apply non-maximum suppression (default is True).
    :param score_threshold: Minimum confidence score required to consider a detection (default is 0.3).
    :param nms_threshold: Intersection over Union (IoU) threshold for suppressing overlapping detections (default is 0.3).
    :param eta: Optional parameter to adjust the NMS threshold dynamically.
    :param top_k: Optional limit on the number of top-scoring detections to consider.
    :param batch_mode: Specifies how to apply NMS in batch scenarios (default is NMSBatchMode.Auto).
    """

    enabled: bool = True
    score_threshold: float = 0.3
    nms_threshold: float = 0.3
    eta: Optional[float] = None
    top_k: Optional[int] = None
    batch_mode: NMSBatchMode = NMSBatchMode.Auto
