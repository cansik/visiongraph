from collections.abc import Callable
from operator import index
from pathlib import Path
from threading import Lock
from time import monotonic_ns
from typing import Any, Generic, Protocol, TypeVar

import cv2
import mediapipe as mp
import numpy as np

from visiongraph.data.Asset import Asset


class _ClosableTask(Protocol):
    def close(self) -> None: ...


_Task = TypeVar("_Task", bound=_ClosableTask)
_Result = TypeVar("_Result")


class MediaPipeTaskRunner(Generic[_Task, _Result]):
    """
    Run a MediaPipe Tasks model in image or video mode.

    Use an options class and the corresponding factory and inference methods from the MediaPipe task. Pass
    the asset and options, including `running_mode`, to `setup()` and `process()`. Call `release()` to close the task.
    """

    def __init__(
        self,
        options_type: Callable[..., Any],
        create: Callable[[Any], _Task],
        detect: Callable[[_Task, mp.Image], _Result],
        detect_for_video: Callable[[_Task, mp.Image, int], _Result],
    ) -> None:
        self._options_type = options_type
        self._create = create
        self._detect = detect
        self._detect_for_video = detect_for_video
        self._task: _Task | None = None
        self._asset: Asset | None = None
        self._options: dict[str, Any] = {}
        self._last_timestamp_ms = -1
        self._lock = Lock()

    def _replace(self, asset: Asset, options: dict[str, Any]) -> _Task:
        previous = self._task
        if previous is not None and self._asset is asset and self._options == options:
            return previous

        # TODO: Use the released in-place Face Landmarker threshold API when available.
        # https://github.com/google-ai-edge/mediapipe/pull/6346
        base_options = mp.tasks.BaseOptions(model_asset_path=str(Path(asset.path).resolve()))
        task = self._create(self._options_type(base_options=base_options, **options))
        self._task = task
        self._asset = asset
        self._options = options
        if previous is not None:
            previous.close()
        return task

    def setup(self, asset: Asset, **options: Any) -> None:
        """Load the model with the specified MediaPipe task options."""
        with self._lock:
            self._replace(asset, options)

    def process(self, image: np.ndarray, asset: Asset, timestamp_ms: int | None = None, **options: Any) -> _Result:
        """
        Process a BGR frame and return the MediaPipe result.

        A changed asset or task option recreates the task and resets its tracking state.

        :param image: The input frame in BGR format.
        :param asset: The model asset.
        :param timestamp_ms: A strictly increasing video timestamp in milliseconds, or None for automatic timestamps.
        :param options: Task options, including the image or video running mode.
        :return: The native MediaPipe task result.
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        with self._lock:
            if self._task is None:
                raise RuntimeError("Set up the MediaPipe estimator before processing frames.")

            if options["running_mode"] == mp.tasks.vision.RunningMode.IMAGE:
                return self._detect(self._replace(asset, options), frame)

            if timestamp_ms is None:
                timestamp_ms = max(monotonic_ns() // 1_000_000, self._last_timestamp_ms + 1)
            else:
                timestamp_ms = index(timestamp_ms)
            if timestamp_ms <= self._last_timestamp_ms:
                raise ValueError("MediaPipe video timestamps must be nonnegative and strictly increasing.")

            task = self._replace(asset, options)
            self._last_timestamp_ms = timestamp_ms
            return self._detect_for_video(task, frame, timestamp_ms)

    def release(self) -> None:
        """Close the task and reset the video timestamps."""
        with self._lock:
            task, self._task = self._task, None
            self._asset = None
            self._options.clear()
            self._last_timestamp_ms = -1
            if task is not None:
                task.close()
