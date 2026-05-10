from pathlib import Path
import sys


def _bootstrap_project_root() -> None:
    root_dir = Path(__file__).resolve().parents[1]
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))


def main() -> None:
    _bootstrap_project_root()
    from tools.model_asset_inventory import iter_config_enums

    config_enums, failed_imports = iter_config_enums()

    current_module = None
    for config_ref in config_enums:
        if config_ref.module_name != current_module:
            current_module = config_ref.module_name
            print(f"- `{current_module}` ()")

        for enum_member in config_ref.enum_class:
            print(f" - {enum_member.name}")

    for failed_import in failed_imports:
        print(f"  [warn] could not import {failed_import.import_name}: {failed_import.error}")


if __name__ == "__main__":
    main()
