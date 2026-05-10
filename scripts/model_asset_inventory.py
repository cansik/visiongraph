import enum
import inspect
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Set, Tuple, Type

from visiongraph import vg
from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset


@dataclass(frozen=True)
class ConfigEnumReference:
    module_name: str
    enum_name: str
    enum_class: Type[enum.Enum]


@dataclass(frozen=True)
class AssetReference:
    module_name: str
    enum_name: str
    member_name: str
    asset_name: str
    asset: Asset


@dataclass(frozen=True)
class FailedImport:
    import_name: str
    module_name: str
    error: str


def iter_config_enums() -> Tuple[Iterable[ConfigEnumReference], Iterable[FailedImport]]:
    config_enums = []
    failed_imports = []
    seen = set()

    for import_name, lazy_import in vg._visiongraph_imports.items():
        if "config" not in import_name.lower():
            continue

        try:
            attribute = lazy_import.attribute
        except Exception as exc:
            failed_imports.append(FailedImport(import_name, lazy_import.module_name, str(exc)))
            continue

        if not inspect.isclass(attribute):
            continue

        if not issubclass(attribute, enum.Enum) or attribute is enum.Enum:
            continue

        key = (attribute.__module__, attribute.__name__)
        if key in seen:
            continue

        seen.add(key)
        config_enums.append(ConfigEnumReference(attribute.__module__, attribute.__name__, attribute))

    config_enums.sort(key=lambda ref: (ref.module_name, ref.enum_name))
    failed_imports.sort(key=lambda failed: (failed.module_name, failed.import_name))
    return config_enums, failed_imports


def iter_assets(value: Any) -> Iterator[Asset]:
    if isinstance(value, Asset):
        yield value
        return

    if isinstance(value, enum.Enum):
        return

    if isinstance(value, dict):
        for item in value.values():
            yield from iter_assets(item)
        return

    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from iter_assets(item)


def collect_asset_references(repository_only: bool = True) -> Tuple[Iterable[AssetReference], Iterable[FailedImport]]:
    config_enums, failed_imports = iter_config_enums()
    references = []
    seen: Set[Tuple[str, str, str, str]] = set()

    for config_ref in config_enums:
        for member in config_ref.enum_class:
            for asset in iter_assets(member.value):
                if repository_only and not isinstance(asset, RepositoryAsset):
                    continue

                asset_name = getattr(asset, "name", repr(asset))
                key = (config_ref.module_name, config_ref.enum_name, member.name, asset_name)
                if key in seen:
                    continue

                seen.add(key)
                references.append(
                    AssetReference(
                        module_name=config_ref.module_name,
                        enum_name=config_ref.enum_name,
                        member_name=member.name,
                        asset_name=asset_name,
                        asset=asset,
                    )
                )

    references.sort(key=lambda ref: (ref.module_name, ref.enum_name, ref.member_name, ref.asset_name))
    return references, failed_imports