import glob
import os

import visiongraph


def reset_data_cache():
    data_path = os.path.abspath(os.path.dirname(visiongraph.data.__file__))
    for file in glob.glob(os.path.join(data_path, "*")):
        if file == __file__:
            continue

        if os.path.isfile(file):
            os.remove(file)
