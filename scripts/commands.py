from pathlib import Path

from setuptools import Command

NAME = "visiongraph"
PACKAGE_NAME = NAME
PACKAGE_VERSION = "1.1.0"
PACKAGE_URL = "https://github.com/cansik/visiongraph"
PACKAGE_DOC_MODULES = ["visiongraph", "!visiongraph.external"]


class GenerateDoc(Command):
    description = "generate pdoc documentation"

    user_options = [
        ("output=", None, "Output path for the documentation."),
        ("launch", None, "Launch webserver to display documentation.")
    ]

    def initialize_options(self):
        self.output: str = "docs"
        self.launch: bool = False

    def finalize_options(self):
        pass

    def run(self) -> None:
        from scripts.import_analyzer import VisiongraphAnalyzer

        # find optional modules
        result = VisiongraphAnalyzer().analyze()

        from scripts.generate_doc import generate_doc
        generate_doc(PACKAGE_NAME, PACKAGE_VERSION, PACKAGE_URL,
                     Path(self.output), PACKAGE_DOC_MODULES, result.optional_modules,
                     launch=bool(self.launch))


class GenerateInitPy(Command):
    description = 'generate top-level init py'
    user_options = []

    def run(self) -> None:
        from scripts.generate_init import generate_init
        generate_init()

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass
