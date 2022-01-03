from pathlib import Path
from typing import List, Set
from sys import platform
from setuptools import setup, find_packages

required_packages = find_packages(exclude=["tests", "examples"])

BASE_NAME = "__required__"
ALL_NAME = "all"


def parse_requirements():
    extras = {}

    with open('requirements.txt') as f:
        lines = f.read().splitlines()

    extra_name = BASE_NAME
    extra_items: List[str] = []

    os_dependent: Set[str] = set()

    skip_extra = False
    for line in [line.strip() for line in lines if line != ""]:
        if line.startswith("# extra"):
            # add current extra
            if not skip_extra:
                extras[extra_name] = extra_items
            else:
                skip_extra = False

            # prepare new extra
            extra_items: List[str] = []

            tokens = line.split(" ")
            extra_name = tokens[2]

            if len(tokens) > 3:
                os_dependent.add(extra_name)
                os_names = tokens[3].split(",")
                if not any([platform.startswith(os_name) for os_name in os_names]):
                    # os not supporting this dependency
                    print(f"Setup: Skipping extra {extra_name} because it is not support on {platform}.")
                    skip_extra = True
                    continue

        elif line.startswith("#"):
            continue
        elif line.startswith("-"):
            continue
        else:
            extra_items.append(line)

    # add last group
    extras[extra_name] = extra_items

    # extract base packages
    install = extras.pop(BASE_NAME)

    # create all group
    all_reqs = [v for k, v in extras.items() if k not in os_dependent]
    extras[ALL_NAME] = []
    for reqs in all_reqs:
        extras[ALL_NAME] += reqs

    return install, extras


install_required, extras_required = parse_requirements()

# read readme
current_dir = Path(__file__).parent
long_description = (current_dir / "README.md").read_text()

setup(
    name="visiongraph",
    version='0.1.9',
    packages=required_packages,
    url='https://github.com/cansik/visiongraph',
    license='MIT License',
    author='Florian Bruggisser',
    author_email='github@broox.ch',
    description='Visiongraph is a high level computer vision pipeline.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=install_required,
    extras_require=extras_required
)
