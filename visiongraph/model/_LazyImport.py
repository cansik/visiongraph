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

        :return: The value of the imported attribute.
        """
        try:
            return self._import()
        except ModuleNotFoundError as ex:
            logging.info(f"Module {self.module_name} not found")

        # create stub to return
        stub = type(self.module_name, _ImportStub.__bases__, dict(_ImportStub.__dict__))
        stub.name = self.module_name
        return stub

    def _import(self) -> Any:
        """
        Imports the module and returns its attribute.

        :return: The value of the imported attribute.
        """
        module = importlib.import_module(self.module_name, package="visiongraph")
        return getattr(module, self.attribute_name)
