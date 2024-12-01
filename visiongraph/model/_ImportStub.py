"""
A mock import stub class to mimic an not imported module.
"""


class _ImportStub:
    """
    An example of a class that raises an ImportError when instantiated.
    """

    name = "NoName"

    def __init__(self) -> None:
        """
        Initializes the _ImportStub object and raises an ImportError.

        :raises ImportError: When the object is instantiated.
        """
        raise ImportError(f"{type(self).name} has not been imported!")
