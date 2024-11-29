import ast
import glob
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Set, DefaultDict, Optional


@dataclass
class Dependency:
    module: str
    name: str
    optional: bool

    @property
    def full_name(self) -> str:
        return f"{self.module}.{self.name}"


@dataclass
class ModuleAnalysisResult:
    dependencies: List[Dependency] = field(default_factory=list)
    optional_modules: List[str] = field(default_factory=list)


class ModuleAnalyzer:
    def __init__(self,
                 root_package: str,
                 excluded_modules: Set[str],
                 late_import_modules: Set[str],
                 optional_modules: Set[str],
                 module_with_methods: Set[str]) -> None:
        self.root_package = root_package
        self.excluded_modules = excluded_modules
        self.late_import_modules = late_import_modules
        self.optional_modules = optional_modules
        self.module_with_methods = module_with_methods

    @staticmethod
    def get_files_in_path(path: str, extensions: Optional[List[str]] = None) -> List[str]:
        if extensions is None:
            extensions = ["*.*"]
        return sorted([f for ext in extensions for f in glob.glob(os.path.join(path, ext), recursive=True)])

    def analyse_source_file(self, file_path: str, dependency_graph: DefaultDict[str, List[str]]) -> List[Dependency]:
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

    def analyze(self) -> ModuleAnalysisResult:
        result = ModuleAnalysisResult()

        # analyze source files
        dependencies: DefaultDict[str, List[str]] = defaultdict(list)
        source_files = self.get_files_in_path(f"{self.root_package}/**", ["*.py"])
        imports = [self.analyse_source_file(f, dependencies) for f in source_files]
        imports = [i for sublist in imports for i in sublist]  # flatten

        # create module to import dict
        imports_dict: DefaultDict[str, List[Dependency]] = defaultdict(list)
        for e in imports:
            imports_dict[e.module].append(e)

        # go through dependencies to find reverse-recursive optional modules
        optional_modules = [m for m in dependencies.keys() if any([m.startswith(e) for e in self.optional_modules])]
        while optional_modules:
            module = optional_modules.pop()
            result.optional_modules.append(module)
            for element in imports_dict[module]:
                element.optional = True
            optional_modules += dependencies[module]

        # unwrap imports dict
        imports = [e for v in imports_dict.values() for e in v]
        imports = sorted(imports, key=lambda x: x.full_name)

        # re-order late imports
        late_imports = [i for i in imports if any([i.module.startswith(e) for e in self.late_import_modules])]
        for li in late_imports:
            imports.remove(li)
            imports.append(li)

        result.dependencies = imports
        return result


class VisiongraphAnalyzer:
    def __init__(self):
        self.root_package = "visiongraph"
        self.excluded_modules = {"visiongraph.external"}
        self.late_import_modules = {
            "visiongraph.estimator.openvino.OpenVinoPoseEstimator",
            "visiongraph.dsp.OneEuroFilterNumba",
            "visiongraph.estimator.spatial.face.landmark.MediaPipeFaceMeshEstimator",
        }
        self.optional_modules = {
            "pyrealsense2",
            "pyk4a",
            "openvino",
            "mediapipe",
            "onnxruntime",
            "moviepy",
            "vidgear",
            "numba",
            "aruco",
            "depthai",
            "faiss",
            "pyzed",
            "glfw",
            "OpenGL",
            "syphon",
            "SpoutGL",
            "filterpy"
        }
        self.module_with_methods = {
            "visiongraph.util",
            "visiongraph.VisionGraphBuilder",
            "visiongraph.estimator"
        }

        self.module_analyzer = ModuleAnalyzer(
            root_package=self.root_package,
            excluded_modules=self.excluded_modules,
            late_import_modules=self.late_import_modules,
            optional_modules=self.optional_modules,
            module_with_methods=self.module_with_methods
        )

    def analyze(self) -> ModuleAnalysisResult:
        return self.module_analyzer.analyze()
