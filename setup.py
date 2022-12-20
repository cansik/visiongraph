import ast
import distutils.cmd
import glob
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from sys import platform
from typing import List
from typing import Set

from setuptools import find_packages
from setuptools import setup


@dataclass
class Dependency:
    module: str
    name: str
    optional: bool

    @property
    def full_name(self) -> str:
        return f"{self.module}.{self.name}"


class GenerateInitPy(distutils.cmd.Command):
    description = 'generate top-level init py'
    user_options = []

    root_package = "visiongraph"

    excluded_modules = {
        "visiongraph.external"
    }

    late_import_modules = {
        "visiongraph.estimator.openvino.OpenVinoPoseEstimator",
        "visiongraph.dsp.OneEuroFilterNumba",
        "visiongraph.estimator.spatial.face.landmark.MediaPipeFaceMeshEstimator",
    }

    optional_modules = {
        "pyrealsense2",
        "pyk4a",
        "openvino",
        "mediapipe",
        "onnxruntime",
        "moviepy",
        "vidgear",
        "numba",
        "aruco",

        # frame buffer sharing
        "glfw",
        "OpenGL",
        "syphonpy",
        "SpoutGL"
    }

    module_with_methods = {
        "visiongraph.util",
        "visiongraph.VisionGraphBuilder",
        "visiongraph.estimator"
    }

    @staticmethod
    def get_files_in_path(path: str, extensions: [str] = ["*.*"]) -> [str]:
        return sorted([f for ext in extensions for f in glob.glob(os.path.join(path, ext), recursive=True)])

    def analyse_source_file(self, file_path: str, dependency_graph) -> List[Dependency]:
        """
        Finds all class definitions in the source files and imports them.
        :param dependency_graph: A dictionary to store dependencies.
        :param file_path: Path to the python source file.
        :return: List of imported modules / names and if they should be optional.
        """
        with open(file_path, "r") as file:
            source = file.read()

        results: List[Dependency] = []

        module = file_path.replace("/", ".").replace("\\", ".").replace(".py", "")
        optional = False

        # skip if is excluded
        if any([module.startswith(e) for e in self.excluded_modules]):
            return results

        nodes = ast.parse(source)
        for node in ast.iter_child_nodes(nodes):
            # classes
            if isinstance(node, ast.ClassDef):
                results.append(Dependency(module, node.name, optional))

            # imports to check if is optional module
            elif isinstance(node, ast.Import):
                for import_name in node.names:
                    dependency_graph[import_name.name].append(module)
                    if any([import_name.name.startswith(e) for e in self.optional_modules]):
                        optional = True

            elif isinstance(node, ast.ImportFrom):
                dependency_graph[node.module].append(module)
                if any([node.module.startswith(e) for e in self.optional_modules]):
                    optional = True

                for import_name in node.names:
                    dependency_graph[import_name.name].append(module)
                    if any([import_name.name.startswith(e) for e in self.optional_modules]):
                        optional = True

            # methods for modules that should be included
            elif isinstance(node, ast.FunctionDef):
                if any([module.startswith(e) for e in self.module_with_methods]):
                    results.append(Dependency(module, node.name, optional))

        # filter private and protected imports
        results = [r for r in results if not r.name.startswith("_")]

        return results

    def run(self) -> None:
        # analyse source files
        dependencies = defaultdict(list)
        source_files = self.get_files_in_path(f"{self.root_package}/**", ["*.py"])
        imports = [self.analyse_source_file(f, dependencies) for f in source_files]
        imports = [i for sl in imports for i in sl]

        # create module to import dict
        imports_dict = defaultdict(list)
        for e in imports:
            imports_dict[e.module].append(e)

        # go through dependencies to find reverse-recursive optional modules
        optional_modules = [m for m in dependencies.keys() if any([m.startswith(e) for e in self.optional_modules])]
        while len(optional_modules) > 0:
            module = optional_modules.pop()
            for element in imports_dict[module]:
                element.optional = True
            optional_modules += dependencies[module]

        # unwrap imports dict
        imports = [e for v in imports_dict.values() for e in v]
        imports = sorted(imports, key=lambda x: x.full_name)

        # re-order-late imports
        # todo: maybe use dependency graph to order by tree-height
        late_imports = [i for i in imports if any([i.module.startswith(e) for e in self.late_import_modules])]
        for li in late_imports:
            imports.remove(li)
            imports.append(li)

        # generate python code
        lines = []
        for imp in imports:
            # relative import
            module = imp.module.replace(self.root_package, "")
            import_line = f"from {module} import {imp.name}"

            if imp.optional:
                line = f"try:\n    {import_line}\nexcept ImportError as ex:\n" \
                       f"    logging.debug(f\"Could not import {imp.name}\")"
                lines.append(line)
            else:
                lines.append(import_line)

        # append header, imports and empty line at the end
        lines.insert(0, "# This file has been auto-generated by setup.py.")
        lines.insert(1, "import logging")
        lines.append("\n")

        # append import stub support
        lines.append("def __getattr__(name):\n"
                     "    from .model._ImportStub import _ImportStub\n"
                     "    logging.debug(f\"{name} has not been imported!\")\n"
                     "    stub = type(name, _ImportStub.__bases__, dict(_ImportStub.__dict__))\n"
                     "    stub.name = name\n"
                     "    return stub")

        lines.append("")

        with open(f"{self.root_package}/__init__.py", "w+") as file:
            file.write("\n".join(lines))

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass


# define required packages
required_packages = find_packages(exclude=["tests", "examples", "tools"])

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
    version="0.1.32.3",
    packages=required_packages,
    url="https://github.com/cansik/visiongraph",
    license="MIT License",
    author="Florian Bruggisser",
    author_email="github@broox.ch",
    description="Visiongraph is a high level computer vision pipeline.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=install_required,
    extras_require=extras_required,
    classifiers=[
        "Development Status :: 3 - Alpha",
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
    },
)
