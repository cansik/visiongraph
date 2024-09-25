from pathlib import Path
from sys import platform
from typing import List
from typing import Set

from setuptools import find_packages
from setuptools import setup

from scripts.generate_doc import GenerateDoc
from scripts.generate_init import GenerateInitPy

# define required packages
required_packages: List[str] = find_packages(exclude=["tests", "examples", "snippets", "assets", "tools"])

BASE_NAME = "__required__"
ALL_NAME = "all"

NAME = "visiongraph"
PACKAGE_NAME = NAME
PACKAGE_VERSION = "1.0.0b3"
PACKAGE_URL = "https://github.com/cansik/visiongraph"
PACKAGE_DOC_MODULES = ["visiongraph", "!visiongraph.external"]


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

# copy values over to generate doc
GenerateDoc.PACKAGE_NAME = PACKAGE_NAME
GenerateDoc.PACKAGE_VERSION = PACKAGE_VERSION
GenerateDoc.PACKAGE_URL = PACKAGE_URL
GenerateDoc.PACKAGE_DOC_MODULES = PACKAGE_DOC_MODULES

setup(
    name=PACKAGE_NAME,
    version=PACKAGE_VERSION,
    packages=required_packages,
    url=PACKAGE_URL,
    license="MIT License",
    author="Florian Bruggisser",
    author_email="github@broox.ch",
    description="Visiongraph is a high level computer vision framework.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=install_required,
    extras_require=extras_required,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    cmdclass={
        "generate_init": GenerateInitPy,
        "doc": GenerateDoc
    },
    entry_points={
        "console_scripts": [
            # "vg-calibrate = tools.CameraCalibratorTool:main",
        ],
    },
)
