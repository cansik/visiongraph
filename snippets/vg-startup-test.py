import time

start = time.time()
from visiongraph import vg

is_first_run = True


def custom(*args):
    global is_first_run
    if is_first_run:
        print(f"It took {(time.time() - start) * 1000:.2f} ms to start visiongraph")
        is_first_run = False


if __name__ == "__main__":
    graph = vg.create_graph(name="VisionGraph", input_node=vg.VideoCaptureInput(), handle_signals=True) \
        .then(vg.custom(custom)) \
        .then(vg.ImagePreview()) \
        .open()
