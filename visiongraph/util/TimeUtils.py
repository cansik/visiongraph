import threading
import time

from visiongraph.util.MathUtils import StreamingMovingAverage


def current_millis() -> int:
    """
    Retrieves the current time in milliseconds.

    :return: Current time in milliseconds since the epoch.
    """
    return time.time_ns() // 1_000_000


class Watch:
    """
    A class to measure elapsed time with start and stop functionalities.
    """

    def __init__(self, name: str = "Watch"):
        """
        Initializes the Watch instance with a name.

        :param name: The name of the watch instance.
        """
        self.name = name
        self.start_time: int = 0
        self.end_time: int = 0
        self.running = False

    def start(self):
        """
        Starts the timer by recording the current time.
        """
        self.start_time = current_millis()
        self.running = True

    def stop(self):
        """
        Stops the timer and records the end time.
        """
        self.end_time = current_millis()
        self.running = False

    def elapsed(self) -> int:
        """
        Calculates the elapsed time since the timer started.

        :return: Elapsed time in milliseconds.
        """
        if self.running:
            return current_millis() - self.start_time
        return self.end_time - self.start_time

    def time_str(self, time_format: str = "%Hh %Mm %Ss {}ms") -> str:
        """
        Formats the elapsed time into a readable string format.

        :param time_format: The format for presenting the time.

        :return: Formatted elapsed time as a string.
        """
        delta = self.elapsed()
        return time.strftime(time_format.format(delta % 1000), time.gmtime(delta / 1000.0))

    def print(self):
        """
        Prints the name of the watch along with the formatted elapsed time.
        """
        print(f"{self.name}: {self.time_str()}")

    def __enter__(self):
        """
        Starts the timer when entering a context.

        :return: The current Watch instance.
        """
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Stops the timer and prints the elapsed time when exiting a context.

        :param exc_type: Exception type.
        :param exc_val: Exception value.
        :param exc_tb: Exception traceback.
        """
        self.stop()
        self.print()


class ProfileWatch(Watch):
    """
    A subclass of Watch that computes the average elapsed time using a moving average.
    """

    def __init__(self, name: str = "Watch", window_size: int = 10):
        """
        Initializes the ProfileWatch instance with a name and window size for moving average.

        :param name: The name of the profile watch instance.
        :param window_size: The size of the window for moving average computation.
        """
        super().__init__(name)
        self._moving_average = StreamingMovingAverage(window_size)

    def stop(self):
        """
        Stops the timer, records the elapsed time, and processes it for the moving average.
        """
        super().stop()
        self._moving_average.process(self.elapsed())

    def average(self):
        """
        Retrieves the current average of the recorded elapsed times.

        :return: The average elapsed time.
        """
        return self._moving_average.average()

    def time_str(self, time_format: str = "%Hh %Mm %Ss {}ms") -> str:
        """
        Formats the elapsed time into a readable string format.

        :param time_format: The format for presenting the time.

        :return: Formatted elapsed time as a string.
        """
        delta = self.elapsed()
        return time.strftime(time_format.format(delta % 1000), time.gmtime(delta / 1000.0))


class FPSTracer:
    """
    A class to track frames per second (FPS) with smoothing options.
    """

    def __init__(self, alpha=0.1):
        """
        Initializes the FPSTracer instance with a smoothing factor.

        :param alpha: The smoothing factor for FPS calculation.
        """
        self.fps = -1
        self.prev_frame_time = 0

        self.alpha = alpha
        self.smooth_fps = -1

    def update(self):
        """
        Updates the FPS based on the time since the last frame and smooths the result.
        """
        current_time = time.time()
        self.fps = 1 / max(0.0001, (current_time - self.prev_frame_time))
        self.prev_frame_time = current_time

        self.smooth_fps += (self.fps - self.smooth_fps) * self.alpha


class HighPrecisionTimer:
    def __init__(self, ensure_monotonic: bool = False):
        """
        Initializes the timer.

        :param ensure_monotonic: If True, ensures the returned timestamps are strictly increasing.
        """
        self.ensure_monotonic = ensure_monotonic
        # Check for the high-resolution performance counter method at initialization.
        if hasattr(time, 'perf_counter_ns'):
            self._use_ns = True
            self._counter = time.perf_counter_ns  # Function returning nanoseconds directly.
        else:
            self._use_ns = False
            # Fallback: convert perf_counter() seconds to nanoseconds.
            self._counter = lambda: int(time.perf_counter() * 1e9)

        # Only create a lock and track the last timestamp if monotonicity is enforced.
        self._lock = threading.Lock() if self.ensure_monotonic else None
        self._last_ns = 0

    def time_ms(self) -> float:
        """
        Returns a high precision timestamp in milliseconds as a float.
        If ensure_monotonic was set to True during initialization, the method
        guarantees that the timestamp will always be strictly increasing.
        """
        current_ns = self._counter()
        if self.ensure_monotonic:
            with self._lock:
                # If the current timestamp is not greater than the last one, bump it.
                if current_ns <= self._last_ns:
                    current_ns = self._last_ns + 1
                self._last_ns = current_ns
        # Convert nanoseconds to milliseconds.
        return current_ns / 1e6
