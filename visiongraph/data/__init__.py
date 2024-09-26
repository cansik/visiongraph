import glob
import os

import visiongraph.cache


def reset_data_cache():
    """
    Removes all files in the data cache directory except for Python files and .gitignore files.

    This function is used to clean up the data cache directory by removing any unnecessary files.
    It ensures that only necessary files, including Python modules, are retained in the cache.
    """
    data_path = os.path.abspath(os.path.dirname(visiongraph.cache.__file__))
    for file in glob.glob(os.path.join(data_path, "*")):
        if file.endswith(".py"):
            continue

        if file.endswith(".gitignore"):
            continue

        if os.path.isfile(file):
            os.remove(file)
