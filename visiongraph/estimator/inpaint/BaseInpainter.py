from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

from visiongraph.estimator.BaseEstimator import BaseEstimator
from visiongraph.result.ImageResult import ImageResult

InpaintInputType = Dict[str, np.ndarray]


class BaseInpainter(BaseEstimator[InpaintInputType, ImageResult], ABC):
    """
    Abstract base class for inpainting estimators.

    Provides a common interface for different inpainting strategies.
    """

    def process(self, data: InpaintInputType) -> ImageResult:
        """
        Processes the input data and returns an image result.

        Args:
            data (InpaintInputType): The input data containing the image to be inpainted.

        Returns:
            ImageResult: The output image after inpainting.
        """
        return self.inpaint(**data)

    @abstractmethod
    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> ImageResult:
        """
        Inpaints the given image using a specific strategy.

        Args:
            image (np.ndarray): The input image to be inpainted.
            mask (np.ndarray): A binary mask indicating the region to be inpainted.

        Returns:
            ImageResult: The output image after inpainting.
        """

        pass
