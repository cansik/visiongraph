import argparse
import importlib.metadata
import sys
from typing import List, Set

import requests
from packaging.requirements import Requirement

# Try to import tomllib (Python 3.11+) or tomli
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        sys.exit("Error: This script requires Python 3.11+ or the 'tomli' package to parse pyproject.toml.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print licenses of dependencies.")
    parser.add_argument(
        "pyproject_file",
        nargs="?",
        default="pyproject.toml",
        help="Path to the pyproject.toml file (default: pyproject.toml)",
    )
    return parser.parse_args()


def extract_requirements_from_list(req_list: List[str]) -> Set[str]:
    requirements = set()
    for line in req_list:
        if not isinstance(line, str):
            continue
        try:
            req = Requirement(line)
            requirements.add(req.name)
        except Exception:
            pass
    return requirements


def parse_pyproject(path: str) -> Set[str]:
    with open(path, "rb") as f:
        data = tomllib.load(f)

    requirements = set()

    project = data.get("project", {})

    # Main dependencies
    requirements.update(extract_requirements_from_list(project.get("dependencies", [])))

    # Optional dependencies
    for group in project.get("optional-dependencies", {}).values():
        requirements.update(extract_requirements_from_list(group))

    # Dependency groups (PEP 735)
    for group in data.get("dependency-groups", {}).values():
        requirements.update(extract_requirements_from_list(group))

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
    pyproject_path = args.pyproject_file

    try:
        requirement_names = parse_pyproject(pyproject_path)
    except FileNotFoundError:
        print(f"Error: File not found: {pyproject_path}")
        return
    except Exception as e:
        print(f"Error parsing {pyproject_path}: {e}")
        return

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
