from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AssetSource:
    name: str
    url: str


@dataclass(frozen=True)
class AssetLicense:
    name: str
    url: Optional[str] = None


@dataclass(frozen=True)
class AssetMetadata:
    source: AssetSource
    license: AssetLicense
    comment: Optional[str] = None

    @classmethod
    def from_values(
        cls,
        source_name: str,
        source_url: str,
        license_name: str,
        license_url: str,
        comment: Optional[str] = None,
    ) -> "AssetMetadata":
        return cls(
            source=AssetSource(name=source_name, url=source_url),
            license=AssetLicense(name=license_name, url=license_url),
            comment=comment,
        )
