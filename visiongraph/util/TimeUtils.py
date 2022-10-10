import time

from visiongraph.util.MathUtils import StreamingMovingAverage


def current_millis() -> int:
    return time.time_ns() // 1_000_000


class Watch:
    def __init__(self, name: str = "Watch"):
        self.name = name
        self.start_time: int = 0
        self.end_time: int = 0
        self.running = False

    def start(self):
        self.start_time = current_millis()
        self.running = True

    def stop(self):
        self.end_time = current_millis()
        self.running = False

    def elapsed(self) -> int:
        if self.running:
            return current_millis() - self.start_time
        return self.end_time - self.start_time

    def time_str(self, time_format: str = "%Hh %Mm %Ss {}ms") -> str:
        delta = self.elapsed()
        return time.strftime(time_format.format(delta % 1000), time.gmtime(delta / 1000.0))

    def print(self):
        print(f"{self.name}: {self.time_str()}")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        self.print()


class ProfileWatch(Watch):
    def __init__(self, name: str = "Watch", window_size: int = 10):
        super().__init__(name)
        self._moving_average = StreamingMovingAverage(window_size)

    def stop(self):
        super().stop()
        self._moving_average.process(self.elapsed())

    def average(self):
        return self._moving_average.average()

    def time_str(self, time_format: str = "%Hh %Mm %Ss {}ms") -> str:
        delta = self.elapsed()
        return time.strftime(time_format.format(delta % 1000), time.gmtime(delta / 1000.0))


class FPSTracer:
    def __init__(self, alpha=0.1):
        self.fps = -1
        self.prev_frame_time = 0

        self.alpha = alpha
        self.smooth_fps = -1

    def update(self):
        # update fps
        current_time = time.time()
        self.fps = 1 / max(0.0001, (current_time - self.prev_frame_time))
        self.prev_frame_time = current_time

        # update smooth
        self.smooth_fps += (self.fps - self.smooth_fps) * self.alpha
