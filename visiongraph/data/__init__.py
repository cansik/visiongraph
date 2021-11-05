import glob
import os

import visiongraph.cache


def reset_data_cache():
    data_path = os.path.abspath(os.path.dirname(visiongraph.cache.__file__))
    for file in glob.glob(os.path.join(data_path, "*")):
        if file.endswith(".py"):
            continue

        if os.path.isfile(file):
            os.remove(file)
