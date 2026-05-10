import argparse
from collections import defaultdict
from pathlib import Path
import sys
from typing import Dict, List, Tuple


def _bootstrap_project_root() -> None:
    root_dir = Path(__file__).resolve().parents[1]
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))


def _collect_asset_references():
    _bootstrap_project_root()
    from tools.model_asset_inventory import collect_asset_references

    return collect_asset_references


def build_markdown() -> str:
    collect_asset_references = _collect_asset_references()
    asset_references, failed_imports = collect_asset_references(repository_only=True)

    grouped_assets: Dict[Tuple[str, str, str, str], List[str]] = defaultdict(list)
    missing_metadata = []

    for asset_reference in asset_references:
        metadata = asset_reference.asset.metadata
        if metadata is None:
            missing_metadata.append(asset_reference)
            continue

        group_key = _metadata_key(metadata)
        grouped_assets[group_key].append(asset_reference.asset_name)

    lines = [
        "# Model Attributions",
        "",
        "This file lists downloadable model artefacts referenced by Visiongraph and their sources and licenses.",
        "",
    ]

    for source_name, source_url, license_name, license_url in sorted(grouped_assets):
        lines.append(f"## {source_name}")
        lines.append("")
        lines.append(f"Origin: {source_url}  ")
        if license_url:
            lines.append(f"License: [{license_name}]({license_url})")
        else:
            lines.append(f"License: {license_name}")
        lines.append("")
        lines.append("Files:")

        for asset_name in sorted(set(grouped_assets[(source_name, source_url, license_name, license_url)])):
            lines.append(f"- {asset_name}")

        lines.append("")

    if missing_metadata:
        lines.append("## Missing Metadata")
        lines.append("")
        lines.append(
            "The following repository assets are referenced by config enums but do not define source/license metadata yet."
        )
        lines.append("")
        for asset_reference in missing_metadata:
            lines.append(
                f"- {asset_reference.asset_name} ({asset_reference.module_name}.{asset_reference.enum_name}.{asset_reference.member_name})"
            )
        lines.append("")

    if failed_imports:
        lines.append("## Skipped Imports")
        lines.append("")
        lines.append("Some config enums could not be imported while generating the attribution list.")
        lines.append("")
        for failed_import in failed_imports:
            lines.append(f"- {failed_import.import_name} ({failed_import.module_name}): {failed_import.error}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _metadata_key(metadata) -> Tuple[str, str, str, str]:
    return (
        metadata.source.name,
        metadata.source.url,
        metadata.license.name,
        metadata.license.url or "",
    )


def main():
    parser = argparse.ArgumentParser(description="Generate a markdown inventory of model attributions.")
    parser.add_argument("--output", type=Path, help="Optional output path. Writes to stdout when omitted.")
    args = parser.parse_args()

    markdown = build_markdown()

    if args.output:
        args.output.write_text(markdown, encoding="utf-8")
        print(f"Wrote {args.output}")
        return

    print(markdown, end="")


if __name__ == "__main__":
    main()
