import importlib
import logging
from dataclasses import dataclass
from typing import Optional, Any

from visiongraph.model._ImportStub import _ImportStub

"""
A dataclass to manage lazy imports in a flexible way.
"""


@dataclass
class _LazyImport:
    """
    A class to represent a lazy import with an attribute.

    :param attribute_name: The name of the attribute to be imported.
    :param module_name: The name of the module to import from.
    :param is_optional: Whether the import is optional. Defaults to False.
    """

    attribute_name: str
    """
    The name of the attribute to be imported.

    Type:
        str
    """

    module_name: str
    """
    The name of the module to import from.

    Type:
        str
    """

    is_optional: bool = False
    """
    Whether the import is optional.

    Type:
        bool
    """

    _attribute: Optional[Any] = None

    @property
    def attribute(self) -> Any:
        """
        Gets or sets the value of the imported attribute.

        :return: The value of the imported attribute.
        """
        if self._attribute is not None:
            return self._attribute

        # import the element
        self._attribute = self._try_import() if self.is_optional else self._import()
        return self._attribute

    def _try_import(self) -> Any:
        """
        Tries to import the module and returns its attribute.

        If the import fails, a stub type is returned that raises an ImportError
        on any usage.

        :return: The value of the imported attribute or an import stub type.
        """
        reason: str | None = None
        try:
            return self._import()
        except ModuleNotFoundError as ex:
            reason = str(ex)
            logging.info(f"Lazy module import {self.module_name} not found (Reason: {reason})")

        # Create a stub class named like the missing module or attribute.
        # Using _ImportStub as a base ensures the metaclass _ImportStubMeta
        # is reused and all traps remain active on the new type.
        stub = type(self.module_name, (_ImportStub,), {})

        stub.name = self.module_name
        stub.reason = reason

        return stub

    def _import(self) -> Any:
        """
        Imports the module and returns its attribute.

        :return: The value of the imported attribute.
        """
        module = importlib.import_module(self.module_name, package="visiongraph")
        return getattr(module, self.attribute_name)
