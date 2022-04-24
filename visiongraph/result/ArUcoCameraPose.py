import numpy as np
import vector

from visiongraph.result.ArUcoMarkerDetection import ArUcoMarkerDetection
from visiongraph.result.BaseResult import BaseResult


class ArUcoCameraPose(BaseResult):
    def __init__(self, position: vector.Vector3D, rotation: vector.Vector3D, marker: ArUcoMarkerDetection):
        self.position = position
        self.rotation = rotation
        self.marker: ArUcoMarkerDetection = marker

    def annotate(self, image: np.ndarray, **kwargs):
        super().annotate(image, **kwargs)
        self.marker.annotate(image, **kwargs)
