from typing import Optional, Sequence

import mediapipe as mp
import numpy as np
import vector

from visiongraph.result.spatial.hand.HandLandmarkResult import HandLandmarkResult


class BlazeHand(HandLandmarkResult):
    def annotate(self, image: np.ndarray, show_info: bool = True, info_text: Optional[str] = None,
                 color: Optional[Sequence[int]] = None,
                 show_bounding_box: bool = False, min_score: float = 0, **kwargs):
        super().annotate(image, show_info, info_text, color, show_bounding_box, min_score,
                         connections=mp.solutions.hands.HAND_CONNECTIONS, **kwargs)

    @property
    def wrist(self) -> vector.Vector4D:
        return self.landmarks[0]

    @property
    def thumb_cmc(self) -> vector.Vector4D:
        return self.landmarks[1]

    @property
    def thumb_mcp(self) -> vector.Vector4D:
        return self.landmarks[2]

    @property
    def thumb_ip(self) -> vector.Vector4D:
        return self.landmarks[3]

    @property
    def thumb_tip(self) -> vector.Vector4D:
        return self.landmarks[4]

    @property
    def index_finger_cmc(self) -> vector.Vector4D:
        return self.landmarks[5]

    @property
    def index_finger_mcp(self) -> vector.Vector4D:
        return self.landmarks[6]

    @property
    def index_finger_ip(self) -> vector.Vector4D:
        return self.landmarks[7]

    @property
    def index_finger_tip(self) -> vector.Vector4D:
        return self.landmarks[8]

    @property
    def middle_finger_cmc(self) -> vector.Vector4D:
        return self.landmarks[9]

    @property
    def middle_finger_mcp(self) -> vector.Vector4D:
        return self.landmarks[10]

    @property
    def middle_finger_ip(self) -> vector.Vector4D:
        return self.landmarks[11]

    @property
    def middle_finger_tip(self) -> vector.Vector4D:
        return self.landmarks[12]

    @property
    def ring_finger_cmc(self) -> vector.Vector4D:
        return self.landmarks[13]

    @property
    def ring_finger_mcp(self) -> vector.Vector4D:
        return self.landmarks[14]

    @property
    def ring_finger_ip(self) -> vector.Vector4D:
        return self.landmarks[15]

    @property
    def ring_finger_tip(self) -> vector.Vector4D:
        return self.landmarks[16]

    @property
    def pinky_cmc(self) -> vector.Vector4D:
        return self.landmarks[17]

    @property
    def pinky_mcp(self) -> vector.Vector4D:
        return self.landmarks[18]

    @property
    def pinky_ip(self) -> vector.Vector4D:
        return self.landmarks[19]

    @property
    def pinky_tip(self) -> vector.Vector4D:
        return self.landmarks[20]
