"""
A mock import stub class to mimic an not imported module.
"""


class _ImportStubMeta(type):
    """
    Metaclass for the _ImportStub class that raises an ImportError
    on class level attribute access.
    """

    def __getattr__(cls, key: str) -> None:
        """
        Raises an ImportError when accessing any missing attribute on
        the stub type.

        This is used for expressions like StubType.some_attribute.

        :param key: The name of the accessed attribute.
        :raises ImportError: Always, because the target module has not been imported.
        """
        raise ImportError(cls._error_message())


class _ImportStub(metaclass=_ImportStubMeta):
    """
    An example of a class that raises an ImportError when instantiated
    or when any of its attributes are used.
    """

    name = "NoName"
    reason: str | None = None

    def __init__(self) -> None:
        """
        Initializes the _ImportStub object and raises an ImportError.

        :raises ImportError: When the object is instantiated.
        """
        raise ImportError(self._error_message())

    def __getattr__(self, key: str) -> None:
        """
        Raises an ImportError when accessing any attribute on the stub instance.

        :param key: The name of the accessed attribute.
        :raises ImportError: Always, because the target module has not been imported.
        """
        raise ImportError(self._error_message())

    def __setattr__(self, key: str, value: object) -> None:
        """
        Raises an ImportError when setting any attribute on the stub instance.

        :param key: The name of the attribute to set.
        :param value: The value that should be assigned.
        :raises ImportError: Always, because the target module has not been imported.
        """
        raise ImportError(self._error_message())

    def __call__(self, *args: object, **kwargs: object) -> None:
        """
        Raises an ImportError when the stub instance is called.

        :raises ImportError: Always, because the target module has not been imported.
        """
        raise ImportError(self._error_message())

    def __getitem__(self, key: object) -> None:
        """
        Raises an ImportError when the stub instance is indexed.

        :param key: The index or key passed to the instance.
        :raises ImportError: Always, because the target module has not been imported.
        """
        raise ImportError(self._error_message())

    @classmethod
    def __class_getitem__(cls, key: object) -> None:
        """
        Raises an ImportError when the stub type is indexed.

        This is used for expressions like StubType["something"].

        :param key: The index or key passed to the type.
        :raises ImportError: Always, because the target module has not been imported.
        """
        raise ImportError(cls._error_message())

    @classmethod
    def _error_message(cls) -> str:
        """
        Builds the error message used by the import stub.

        :return: The error message describing which module was not imported.
        """
        main_message = f"{cls.name} has not been imported!"
        reason_message = f" Reason: {cls.reason}" if cls.reason else ""
        return f"{main_message}{reason_message}"
