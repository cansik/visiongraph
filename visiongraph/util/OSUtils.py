import sys


def isMacOSX() -> bool:
    """
    Checks if the current operating system is macOS.

    Returns:
        bool: True if the operating system is macOS, False otherwise.
    """
    return sys.platform == "darwin"


def isWindows() -> bool:
    """
    Checks if the current operating system is Windows.

    Returns:
        bool: True if the operating system is Windows, False otherwise.
    """
    return sys.platform == "win32"


def isLinux() -> bool:
    """
    Checks if the current operating system is Linux.

    Returns:
        bool: True if the operating system is Linux, False otherwise.
    """
    return sys.platform == "linux"
