import importlib
import unittest
from functools import partial
from unittest import mock

from visiongraph.model._ImportStub import _ImportStub
from visiongraph.model._LazyImport import _LazyImport


class ImportStubTests(unittest.TestCase):
    def _make_stub_type(self, name="MyMissingType", reason="missing dependency"):
        stub_type = type(name, (_ImportStub,), {})
        stub_type.name = name
        stub_type.reason = reason
        return stub_type

    def test_error_message_contains_name_and_reason(self):
        stub_type = self._make_stub_type(name="SomeType", reason="backend not installed")
        message = stub_type._error_message()

        self.assertIn("SomeType has not been imported", message)
        self.assertIn("backend not installed", message)

    def test_instantiation_raises_import_error(self):
        stub_type = self._make_stub_type()
        with self.assertRaises(ImportError):
            stub_type()

    def test_instance_attribute_access_raises_import_error(self):
        stub_type = self._make_stub_type()
        instance = object.__new__(stub_type)

        with self.assertRaises(ImportError):
            _ = instance.some_attribute

    def test_class_attribute_access_raises_import_error(self):
        stub_type = self._make_stub_type()

        with self.assertRaises(ImportError):
            _ = stub_type.create

    def test_class_index_access_raises_import_error(self):
        stub_type = self._make_stub_type()

        with self.assertRaises(ImportError):
            _ = stub_type["anything"]

    def test_partial_on_missing_method_raises_import_error(self):
        stub_type = self._make_stub_type()

        with self.assertRaises(ImportError):
            partial(stub_type.create, "info")


class LazyImportTests(unittest.TestCase):
    def test_import_existing_module_attribute(self):
        lazy = _LazyImport(attribute_name="sqrt", module_name="math")
        sqrt = lazy.attribute

        self.assertEqual(sqrt(9), 3.0)

    def test_attribute_is_cached_after_first_access(self):
        lazy = _LazyImport(attribute_name="sqrt", module_name="math")

        real_import_module = importlib.import_module
        with mock.patch("importlib.import_module") as mocked_import_module:
            mocked_import_module.side_effect = real_import_module

            first = lazy.attribute
            second = lazy.attribute

            self.assertIs(first, second)
            mocked_import_module.assert_called_once_with("math", package="visiongraph")

    def test_optional_missing_module_returns_stub_type(self):
        missing_module = "this_module_does_not_exist_for_tests_12345"
        lazy = _LazyImport(
            attribute_name="SomeAttribute",
            module_name=missing_module,
            is_optional=True,
        )

        stub = lazy.attribute

        self.assertIsInstance(stub, type)
        self.assertTrue(issubclass(stub, _ImportStub))
        self.assertEqual(stub.name, missing_module)
        self.assertIsNotNone(stub.reason)
        self.assertIn(missing_module, stub.reason)

    def test_optional_missing_module_stub_usage_raises_import_error(self):
        missing_module = "another_missing_module_for_tests_67890"
        lazy = _LazyImport(
            attribute_name="SomeAttribute",
            module_name=missing_module,
            is_optional=True,
        )

        stub = lazy.attribute

        with self.assertRaises(ImportError):
            _ = stub.create

        with self.assertRaises(ImportError):
            _ = stub["anything"]

        with self.assertRaises(ImportError):
            partial(stub.create, "info")

        with self.assertRaises(ImportError):
            stub()

    def test_required_missing_module_raises_module_not_found_error(self):
        missing_module = "required_missing_module_for_tests_54321"
        lazy = _LazyImport(
            attribute_name="SomeAttribute",
            module_name=missing_module,
            is_optional=False,
        )

        with self.assertRaises(ModuleNotFoundError):
            _ = lazy.attribute


if __name__ == "__main__":
    unittest.main()
