from typing import Tuple, Any, Union

import numpy as np


class RollingBufferNumpy:
    def __init__(self, buffer_size: int, feature_shape: Union[int, Tuple[int, ...]], dtype: Any = np.float32):
        """
        Initialize a rolling buffer with a fixed block size and customizable data type.

        :param buffer_size: Number of samples in the buffer.
        :param feature_shape: Shape of each feature sample (e.g., landmark dimensions).
        :param dtype: Data type of the buffer elements (default: np.float32).
        """
        if isinstance(feature_shape, int):
            feature_shape = feature_shape,

        self.buffer_size = buffer_size
        self.feature_shape = feature_shape
        self.dtype = dtype
        self.buffer = np.zeros((buffer_size, *feature_shape), dtype=dtype)
        self.index = 0

    def add(self, sample: np.ndarray):
        """
        Add a new sample to the rolling buffer, replacing the oldest sample if the buffer is full.

        :param sample: New sample to add, must match the feature shape and data type.
        """
        if sample.shape != self.feature_shape:
            raise ValueError(f"Sample shape {sample.shape} does not match buffer feature shape {self.feature_shape}.")
        if sample.dtype != self.dtype:
            raise ValueError(f"Sample dtype {sample.dtype} does not match buffer dtype {self.dtype}.")

        self.buffer[self.index] = sample
        self.index = (self.index + 1) % self.buffer_size

    def get(self) -> np.ndarray:
        """
        Get the current state of the rolling buffer.

        :return: A view of the buffer with the most recent samples in order.
        """
        return np.roll(self.buffer, -self.index, axis=0)

    def reset(self):
        """
        Reset the rolling buffer to its initial state.
        """
        self.buffer = np.zeros((self.buffer_size, *self.feature_shape), dtype=self.dtype)
        self.index = 0
