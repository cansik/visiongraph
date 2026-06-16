import argparse
import hashlib
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urlparse

import requests

from scripts.model_asset_inventory import AssetReference, FailedImport, collect_asset_references


def build_download_plan() -> Tuple[Dict[Tuple[str, str, str, str], List[AssetReference]], List[FailedImport]]:
    asset_references, failed_imports = collect_asset_references(repository_only=True)
    grouped_assets: Dict[Tuple[str, str, str, str], List[AssetReference]] = defaultdict(list)

    for asset_reference in asset_references:
        metadata = asset_reference.asset.metadata
        if metadata is None:
            raise ValueError(
                "Cannot download model licenses because repository asset metadata is missing for "
                f"{asset_reference.module_name}.{asset_reference.enum_name}.{asset_reference.member_name}: "
                f"{asset_reference.asset_name}"
            )

        if not metadata.license.url:
            raise ValueError(
                "Cannot download model licenses because a license URL is missing for "
                f"{asset_reference.module_name}.{asset_reference.enum_name}.{asset_reference.member_name}: "
                f"{asset_reference.asset_name}"
            )

        group_key = (
            metadata.source.name,
            metadata.source.url,
            metadata.license.name,
            metadata.license.url,
        )
        grouped_assets[group_key].append(asset_reference)

    return grouped_assets, list(failed_imports)


def resolve_download_url(url: str) -> str:
    parsed = urlparse(url)

    if parsed.netloc == "github.com":
        path_parts = [part for part in parsed.path.split("/") if part]
        if len(path_parts) >= 5 and path_parts[2] == "blob":
            owner, repo, _blob, ref = path_parts[:4]
            remainder = "/".join(path_parts[4:])
            return f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{remainder}"

    return url


def build_output_filename(source_name: str, license_name: str, license_url: str) -> str:
    slug = _slugify(f"{source_name}-{license_name}") or "license"
    url_hash = hashlib.sha256(license_url.encode("utf-8")).hexdigest()[:8]
    suffix = _preferred_suffix(license_url)
    return f"{slug}-{url_hash}{suffix}"


def download_licenses(output_dir: Path) -> List[Path]:
    grouped_assets, failed_imports = build_download_plan()
    if failed_imports:
        formatted_failures = "\n".join(
            f"- {failed_import.import_name} ({failed_import.module_name}): {failed_import.error}"
            for failed_import in failed_imports
        )
        raise RuntimeError(
            f"Cannot download model licenses because some config enums could not be imported:\n{formatted_failures}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    written_files = []
    for source_name, _source_url, license_name, license_url in sorted(grouped_assets):
        download_url = resolve_download_url(license_url)
        response = requests.get(download_url, timeout=30)
        response.raise_for_status()

        output_path = output_dir / build_output_filename(source_name, license_name, license_url)
        output_path.write_text(response.text, encoding="utf-8")
        written_files.append(output_path)

    return written_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download model license files into a local directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/licenses"),
        help="Directory where downloaded license files are written (default: build/licenses)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written_files = download_licenses(args.output_dir)
    for written_file in written_files:
        print(f"Wrote {written_file}")


def _preferred_suffix(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".txt", ".md", ".rst", ".html"}:
        return suffix
    return ".txt"


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


if __name__ == "__main__":
    main()
