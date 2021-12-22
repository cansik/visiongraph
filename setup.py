from typing import List
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
    all_reqs = list(extras.values())
    extras[ALL_NAME] = []
    for reqs in all_reqs:
        extras[ALL_NAME] += reqs

    return install, extras


install_required, extras_required = parse_requirements()

setup(
    name="visiongraph",
    version='0.1.9',
    packages=required_packages,
    url='https://github.com/cansik/visiongraph',
    license='MIT License',
    author='Florian Bruggisser',
    author_email='github@broox.ch',
    description='Visiongraph is a computer vision pipeline.',
    install_requires=install_required,
    extras_require=extras_required
)
