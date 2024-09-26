from typing import Optional, Sequence

import mediapipe as mp
import numpy as np
import vector

from visiongraph.result.spatial.hand.HandLandmarkResult import HandLandmarkResult


class BlazeHand(HandLandmarkResult):
    """
    A class representing a hand landmark result with additional properties.
    """

    def annotate(self, image: np.ndarray, show_info: bool = True, info_text: Optional[str] = None,
                 color: Optional[Sequence[int]] = None,
                 show_bounding_box: bool = False, min_score: float = 0, **kwargs):
        """
        Annotates the given image with hand landmarks.

        Args:
            image (np.ndarray): The input image.
            show_info (bool, optional): Whether to display information. Defaults to True.
            info_text (Optional[str], optional): Information text to be displayed. Defaults to None.
            color (Optional[Sequence[int]], optional): Color to be used for annotation. Defaults to None.
            show_bounding_box (bool, optional): Whether to display the bounding box. Defaults to False.
            min_score (float, optional): Minimum score for hand landmarks. Defaults to 0.

        Returns:
            bool: Whether the annotation was successful.
        """
        return super().annotate(image, show_info, info_text, color, show_bounding_box, min_score,
                                connections=mp.solutions.hands.HAND_CONNECTIONS, **kwargs)

    @property
    def wrist(self) -> vector.Vector4D:
        """
        Gets the position of the wrist.

        Returns:
            vector.Vector4D: The position of the wrist.
        """
        return self.landmarks[0]

    @property
    def thumb_cmc(self) -> vector.Vector4D:
        """
        Gets the position of the thumb CMC (Distal Interphalangeal joint).

        Returns:
            vector.Vector4D: The position of the thumb CMC.
        """
        return self.landmarks[1]

    @property
    def thumb_mcp(self) -> vector.Vector4D:
        """
        Gets the position of the thumb MCP (Metacarpophalangeal joint).

        Returns:
            vector.Vector4D: The position of the thumb MCP.
        """
        return self.landmarks[2]

    @property
    def thumb_ip(self) -> vector.Vector4D:
        """
        Gets the position of the thumb IP (Interphalangeal joint).

        Returns:
            vector.Vector4D: The position of the thumb IP.
        """
        return self.landmarks[3]

    @property
    def thumb_tip(self) -> vector.Vector4D:
        """
        Gets the position of the thumb tip.

        Returns:
            vector.Vector4D: The position of the thumb tip.
        """
        return self.landmarks[4]

    @property
    def index_finger_cmc(self) -> vector.Vector4D:
        """
        Gets the position of the index finger CMC (Distal Interphalangeal joint).

        Returns:
            vector.Vector4D: The position of the index finger CMC.
        """
        return self.landmarks[5]

    @property
    def index_finger_mcp(self) -> vector.Vector4D:
        """
        Gets the position of the index finger MCP (Metacarpophalangeal joint).

        Returns:
            vector.Vector4D: The position of the index finger MCP.
        """
        return self.landmarks[6]

    @property
    def index_finger_ip(self) -> vector.Vector4D:
        """
        Gets the position of the index finger IP (Interphalangeal joint).

        Returns:
            vector.Vector4D: The position of the index finger IP.
        """
        return self.landmarks[7]

    @property
    def index_finger_tip(self) -> vector.Vector4D:
        """
        Gets the position of the index finger tip.

        Returns:
            vector.Vector4D: The position of the index finger tip.
        """
        return self.landmarks[8]

    @property
    def middle_finger_cmc(self) -> vector.Vector4D:
        """
        Gets the position of the middle finger CMC (Distal Interphalangeal joint).

        Returns:
            vector.Vector4D: The position of the middle finger CMC.
        """
        return self.landmarks[9]

    @property
    def middle_finger_mcp(self) -> vector.Vector4D:
        """
        Gets the position of the middle finger MCP (Metacarpophalangeal joint).

        Returns:
            vector.Vector4D: The position of the middle finger MCP.
        """
        return self.landmarks[10]

    @property
    def middle_finger_ip(self) -> vector.Vector4D:
        """
        Gets the position of the middle finger IP (Interphalangeal joint).

        Returns:
            vector.Vector4D: The position of the middle finger IP.
        """
        return self.landmarks[11]

    @property
    def middle_finger_tip(self) -> vector.Vector4D:
        """
        Gets the position of the middle finger tip.

        Returns:
            vector.Vector4D: The position of the middle finger tip.
        """
        return self.landmarks[12]

    @property
    def ring_finger_cmc(self) -> vector.Vector4D:
        """
        Gets the position of the ring finger CMC (Distal Interphalangeal joint).

        Returns:
            vector.Vector4D: The position of the ring finger CMC.
        """
        return self.landmarks[13]

    @property
    def ring_finger_mcp(self) -> vector.Vector4D:
        """
        Gets the position of the ring finger MCP (Metacarpophalangeal joint).

        Returns:
            vector.Vector4D: The position of the ring finger MCP.
        """
        return self.landmarks[14]

    @property
    def ring_finger_ip(self) -> vector.Vector4D:
        """
        Gets the position of the ring finger IP (Interphalangeal joint).

        Returns:
            vector.Vector4D: The position of the ring finger IP.
        """
        return self.landmarks[15]

    @property
    def ring_finger_tip(self) -> vector.Vector4D:
        """
        Gets the position of the ring finger tip.

        Returns:
            vector.Vector4D: The position of the ring finger tip.
        """
        return self.landmarks[16]

    @property
    def pinky_cmc(self) -> vector.Vector4D:
        """
        Gets the position of the pinky CMC (Distal Interphalangeal joint).

        Returns:
            vector.Vector4D: The position of the pinky CMC.
        """
        return self.landmarks[17]

    @property
    def pinky_mcp(self) -> vector.Vector4D:
        """
        Gets the position of the pinky MCP (Metacarpophalangeal joint).

        Returns:
            vector.Vector4D: The position of the pinky MCP.
        """
        return self.landmarks[18]

    @property
    def pinky_ip(self) -> vector.Vector4D:
        """
        Gets the position of the pinky IP (Interphalangeal joint).

        Returns:
            vector.Vector4D: The position of the pinky IP.
        """
        return self.landmarks[19]

    @property
    def pinky_tip(self) -> vector.Vector4D:
        """
        Gets the position of the pinky tip.

        Returns:
            vector.Vector4D: The position of the pinky tip.
        """
        return self.landmarks[20]
