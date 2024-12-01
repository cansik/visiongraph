from typing import List

import numpy as np

from visiongraph.external.intel.models.model import Model


class SyncInferencePipeline:
    """
    A class representing a synchronized inference pipeline for the given model.
    """

    def __init__(self, model: Model, device: str = "AUTO"):
        """
        Initializes the SyncInferencePipeline object.

        :param model: The model to be used in the pipeline.
        :param device: The target device. Defaults to "AUTO".
        """
        self.device = device
        self.model = model

    def setup(self):
        """
        Loads the model if it's not already loaded.

        """
        if not self.model.model_loaded:
            self.model.load()

    def process(self, data: np.ndarray) -> List:
        """
        Processes the given input data through the synchronized inference pipeline.

        :param data: The input data to be processed.

        :return: A list of output results.
        """
        inputs, preprocessing_meta = self.model.preprocess(data)
        raw_result = self.model.infer_sync(inputs)
        outputs = self.model.postprocess(raw_result, preprocessing_meta)
        return outputs

    def release(self):
        """
        Releases the pipeline resources.

        """
        pass
