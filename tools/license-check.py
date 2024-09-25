import argparse
import importlib.metadata
from typing import List, Set

import requests
from packaging.requirements import Requirement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print licenses of dependencies.")
    parser.add_argument(
        "requirements_file",
        nargs="?",
        default="requirements.txt",
        help="Path to the requirements.txt file (default: requirements.txt)",
    )
    return parser.parse_args()


def parse_requirements(lines: List[str]) -> Set[str]:
    requirements = set()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            req = Requirement(line)
            requirements.add(req.name)
        except Exception:
            pass  # Skip invalid requirement lines
    return requirements


def normalize_license(license_str: str) -> str:
    license_str_lower = license_str.lower()
    license_map = {
        "mit license": "MIT License",
        "mit": "MIT License",
        "bsd license": "BSD License",
        "bsd": "BSD License",
        "apache license 2.0": "Apache License 2.0",
        "apache": "Apache License 2.0",
        "apache 2.0": "Apache License 2.0",
        "apache license, version 2.0": "Apache License 2.0",
        "gnu general public license v3 (gplv3)": "GPLv3",
        "gplv3": "GPLv3",
        "gpl": "GPL",
        "lgpl": "LGPL",
        "mozilla public license 2.0": "MPL 2.0",
        "mpl 2.0": "MPL 2.0",
        "python software foundation license": "PSF License",
        "psf": "PSF License",
        "artistic license": "Artistic License",
        "zlib/libpng license": "Zlib/Libpng License",
        "isc license": "ISC License",
        "academic free license (afl)": "AFL",
        "afl": "AFL",
        "open software license": "OSL",
        "osl": "OSL",
        "eclipse public license": "EPL",
        "epl": "EPL",
        "unknown": "Unknown",
        "public domain": "Public Domain",
    }
    for key, value in license_map.items():
        if key in license_str_lower:
            return value
    return "Unknown"


def get_license(package_name: str) -> str:
    try:
        metadata = importlib.metadata.metadata(package_name)
        license_str = metadata.get("License", "")
        if not license_str or license_str == "UNKNOWN":
            classifiers = metadata.get_all("Classifier") or []
            for classifier in classifiers:
                if classifier.startswith("License"):
                    license_str = classifier
                    break
        if not license_str or license_str == "UNKNOWN":
            return get_license_from_pypi(package_name)
        return normalize_license(license_str)
    except importlib.metadata.PackageNotFoundError:
        return get_license_from_pypi(package_name)


def get_license_from_pypi(package_name: str) -> str:
    url = f"https://pypi.org/pypi/{package_name}/json"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            license_str = data["info"].get("license", "")
            if not license_str or license_str == "UNKNOWN":
                classifiers = data["info"].get("classifiers", [])
                for classifier in classifiers:
                    if classifier.startswith("License"):
                        license_str = classifier
                        break
            return normalize_license(license_str)
        else:
            return "Unknown"
    except Exception:
        return "Unknown"


def main() -> None:
    args = parse_args()
    requirements_path = args.requirements_file
    with open(requirements_path, "r") as f:
        lines = f.readlines()
    requirement_names = parse_requirements(lines)
    packages = sorted(requirement_names, key=lambda s: s.lower())

    # Calculate the maximum length for alignment
    max_name_length = max(len(name) for name in packages) if packages else 0

    # Print header
    print(f"{'Package'.ljust(max_name_length)}  License")
    print(f"{'-' * max_name_length}  {'-' * 20}")

    for name in packages:
        license = get_license(name)
        print(f"{name.ljust(max_name_length)}  {license}")


if __name__ == "__main__":
    main()
