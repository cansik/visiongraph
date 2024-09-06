import importlib
import logging
from dataclasses import dataclass
from typing import Optional, Any

from visiongraph.model._ImportStub import _ImportStub


@dataclass
class _LazyImport:
    attribute_name: str
    module_name: str
    is_optional: bool = False

    _attribute: Optional[Any] = None

    @property
    def attribute(self) -> Any:
        if self._attribute is not None:
            return self._attribute

        # import the element
        self._attribute = self._try_import() if self.is_optional else self._import()
        return self._attribute

    def _try_import(self) -> Any:
        try:
            return self._import()
        except ModuleNotFoundError as ex:
            logging.info(f"Module {self.module_name} not found")

        # create stub to return
        stub = type(self.module_name, _ImportStub.__bases__, dict(_ImportStub.__dict__))
        stub.name = self.module_name
        return stub

    def _import(self) -> Any:
        module = importlib.import_module(self.module_name, package="visiongraph")
        return getattr(module, self.attribute_name)
