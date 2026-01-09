import inspect
import importlib
import enum
from inspect import isclass

from visiongraph import vg

# object detection models and configurations

for name, item in vg._visiongraph_imports.items():
    cls = item.attribute
    if not isclass(cls):
        continue
    if issubclass(cls, vg.ObjectDetector):
        if inspect.isabstract(cls):
            continue

        module_name = item.attribute.__module__
        print(f"- `{module_name}` ()")

        try:
            mod = importlib.import_module(module_name)
        except Exception as e:
            print(f"  [warn] could not import module: {e}")
            continue

        enums_found = []
        for attr_name, attr_value in inspect.getmembers(mod):
            if attr_name.startswith("_"):
                continue
            if inspect.isclass(attr_value) and issubclass(attr_value, enum.Enum) and attr_value is not enum.Enum:
                member_names = [member.name for member in attr_value]
                enums_found.append((attr_name, member_names))

        if not enums_found:
            continue
        else:
            for enum_name, member_names in enums_found:
                if "config" not in enum_name.lower():
                    continue

                # print(f' - {enum_name}')

                for enum_member in member_names:
                    print(f" - {enum_member}")
