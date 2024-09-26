import numpy as np

from visiongraph.result.BaseResult import BaseResult


class ImageResult(BaseResult):
    """
    A result class for image processing tasks.
    """

    def __init__(self, output: np.ndarray) -> None:
        """
        Initializes the ImageResult object with the given output image.

        Args:
            output (np.ndarray): The output image to be stored in this result.
        """
        self.output = output

    def annotate(self, image: np.ndarray, **kwargs) -> None:
        """
        Annotates the given image by drawing the output of this result onto it.

        This method calls the parent class's `annotate` method and then draws
        the output image onto the input image. The implementation of drawing is
        currently left as a TODO.

        Args:
            image (np.ndarray): The input image to be annotated.
            **kwargs: Additional keyword arguments to be passed to the parent
                class's `annotate` method.
        """
        super().annotate(image, **kwargs)

        # todo: implement drawing image onto other image
