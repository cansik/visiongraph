from abc import ABC, abstractmethod
from typing import List, TypeVar

from visiongraph.estimator.ScoreThresholdEstimator import ScoreThresholdEstimator
from visiongraph.result.ClassificationResult import ClassificationResult

InputType = TypeVar('InputType')
OutputType = TypeVar('OutputType', bound=ClassificationResult)


class BaseClassifier(ScoreThresholdEstimator[InputType, OutputType], ABC):
    """
    A base class for classification estimators that inherit the score thresholding functionality.
    """

    def __init__(self, min_score: float):
        """
        Initializes the BaseClassifier with a specified minimum score.

        Args:
            min_score (float): The minimum score required to classify a sample as positive.
        """
        super().__init__(min_score)
        self.labels: List[str] = []

    @abstractmethod
    def process(self, data: InputType) -> OutputType:
        """
        Processes the input data and returns a classification result.

        Args:
            data (InputType): The input data to be processed.

        Returns:
            OutputType: A ClassificationResult object containing the predicted label and score.
        """
        pass

    def get_label(self, index: int) -> str:
        """
        Retrieves the label at the specified index. If the index is out of range, returns a string representation of the index.

        Args:
            index (int): The index of the label to be retrieved.

        Returns:
            str: The label at the specified index or a string representation of the index.
        """
        if 0 <= index < len(self.labels):
            return self.labels[index]
        return str(index)
