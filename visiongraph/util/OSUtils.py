import sys


def isMacOSX() -> bool:
    return sys.platform == "darwin"


def isWindows() -> bool:
    return sys.platform == "win32"


def isLinux() -> bool:
    return sys.platform == "linux"
