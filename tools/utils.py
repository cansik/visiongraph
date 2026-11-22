import glob
import os
from collections.abc import Iterable


def get_files_in_path(path: str, extensions: Iterable[str] = ("*.*",), recursive: bool = True) -> list[str]:
    return sorted([f for ext in extensions for f in glob.glob(os.path.join(path, ext), recursive=recursive)])
