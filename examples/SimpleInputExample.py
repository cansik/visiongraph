import visiongraph as vg

if __name__ == "__main__":
    graph = vg.create_graph(name="VisionGraph", input_node=vg.AzureKinectInput(), handle_signals=True) \
        .then(vg.ImagePreview()) \
        .open()
