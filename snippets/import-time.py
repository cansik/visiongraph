import time

if False:
    start = time.time()
    import visiongraph as vg

    end = time.time()

    print(f"It took {(end - start) * 1000:.2f} ms to import visiongraph")

# attribute_names = list(vg._visiongraph_imports.keys())
attribute_names = ['AsyncGraphNode', 'BaseGraph', 'GraphNode', 'Processable', 'VisionGraph', 'add_breakpoint',
                   'create_graph', 'custom', 'extract', 'passthrough', 'sequence', 'Asset', 'LocalAsset',
                   'RepositoryAsset', 'BaseFilterNumpy', 'LandmarkSmoothFilter', 'OneEuroFilter', 'OneEuroFilterNumpy',
                   'VectorNumpySmoothFilter', 'BaseClassifier', "AdasFaceConfig", "AdasFaceDetector", "FaceDetector",
                   "OpenVinoFaceConfig", "OpenVinoFaceDetector", "AffectNetEmotionClassifier"]

times = []

print("testing...")
for r in range(100000):
    for i, name in enumerate(attribute_names):
        start = time.time()
        # access element
        import visiongraph as vg

        e = vg.__getattribute__(name)
        end = time.time()
        times.append(end - start)

average_time = sum(times) / len(times)

print(f"Average Time: {average_time * 1000:2f} ms")

# without lazy import:            Average Time: 0.000762 ms
# with lazy import (dict lookup): Average Time: 0.000636 ms
# with lazy import (attribute):
