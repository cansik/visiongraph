from visiongraph import vg

if __name__ == "__main__":
    graph = (
        vg.create_graph(name="VisionGraph", input_node=vg.VideoCaptureInput(), handle_signals=True)
        .then(vg.ImagePreview())
        .open()
    )
