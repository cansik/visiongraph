import sys
from dataclasses import dataclass
from pathlib import Path

from setuptools import Command


@dataclass
class ProjectConfig:
    name: str
    version: str
    url: str | None
    doc_modules: list[str]


def load_project_config(path: Path = Path("pyproject.toml")) -> ProjectConfig:
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib

    with path.open("rb") as f:
        data = tomllib.load(f)

    project = data["project"]
    urls = project.get("urls", {})
    name = project["name"]

    return ProjectConfig(
        name=name,
        version=project["version"],
        url=urls.get("Homepage"),
        doc_modules=[name, f"!{name}.external"],
    )


class GenerateDoc(Command):
    description = "generate pdoc documentation"

    user_options: list[tuple[str, str | None, str]] = [
        ("output=", None, "Output path for the documentation."),
        ("launch", None, "Launch webserver to display documentation."),
    ]

    def initialize_options(self) -> None:
        self.output: str = "docs"
        self.launch: bool = False

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        from scripts.generate_doc import generate_doc
        from scripts.import_analyzer import VisiongraphAnalyzer

        config = load_project_config()

        # find optional modules
        result = VisiongraphAnalyzer().analyze()

        generate_doc(
            config.name,
            config.version,
            config.url,
            Path(self.output),
            config.doc_modules,
            result.optional_modules,
            launch=bool(self.launch),
        )


class GenerateInitPy(Command):
    description = "generate top-level init py"
    user_options: list[tuple[str, str | None, str]] = []

    def run(self) -> None:
        from scripts.generate_init import generate_init

        generate_init()

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass
