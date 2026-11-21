import argparse
import os
import re
import shutil
import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Union, Optional, List, Sequence
from unittest.mock import MagicMock

import pdoc
import pdoc.web
from jinja2 import pass_context
from jinja2.runtime import Context
from markupsafe import Markup
from pdoc.render_helpers import qualname_candidates, possible_sources, relative_link

from scripts import pdoc_monkeypatch


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

    with path.open("rb") as file:
        data = tomllib.load(file)

    project = data["project"]
    urls = project.get("urls", {})
    name = project["name"]

    return ProjectConfig(
        name=name,
        version=project["version"],
        url=urls.get("Homepage"),
        doc_modules=[name, f"!{name}.external"],
    )


class AutoMock(MagicMock):
    """Supply stand-ins for attributes of an unavailable optional dependency."""

    def __getattr__(self, name):
        # Avoid recursion for internal MagicMock attributes
        if name.startswith("_"):
            return super().__getattr__(name)

        # For any other attribute access, return another AutoMock
        return AutoMock(name=name)


class OptionalModuleStub(ModuleType):
    """Allow pdoc to import modules that use an unavailable optional dependency."""

    def __init__(self, name: str):
        super().__init__(name)
        self.__path__: list[str] = []

    def __getattr__(self, name: str) -> AutoMock:
        if name.startswith("_"):
            raise AttributeError(name)
        return AutoMock(name=name)


def stub_unavailable_imports(optional_imports: Sequence[str]) -> None:
    module_names: set[str] = set()
    for name in optional_imports:
        while name:
            module_names.add(name)
            name = name.rpartition(".")[0]

    for name in sorted(module_names):
        if name in sys.modules:
            continue
        try:
            import_module(name)
        except ImportError:
            stub = OptionalModuleStub(name)
            sys.modules[name] = stub
            parent, _, member = name.rpartition(".")
            if parent:
                setattr(sys.modules[parent], member, stub)


@pass_context
def custom_linkify(context: Context, code: str, namespace: str = "") -> str:
    """
    Link all identifiers in a block of text. Identifiers referencing unknown modules or modules that
    are not rendered at the moment will be ignored.
    A piece of text is considered to be an identifier if it either contains a `.` or is surrounded by `<code>` tags.
    """

    def linkify_repl(m: re.Match):
        text = m.group(0)
        plain_text = text.replace('</span><span class="o">.</span><span class="n">', ".")
        identifier = plain_text.removesuffix("()")
        mod: pdoc.doc.Module = context["module"]

        # Check if this is a relative reference?
        if identifier.startswith("."):
            taken_from_mod = mod
            if namespace and (ns := mod.get(namespace)):
                # Imported from somewhere else, so the relative reference should be from the original module.
                taken_from_mod = context["all_modules"].get(ns.taken_from[0], mod)
            if taken_from_mod.is_package:
                # If we are in __init__.py, we want `.foo` to refer to a child module.
                parent_module = taken_from_mod.modulename
            else:
                # If we are in a leaf module, we want `.foo` to refer to the adjacent module.
                parent_module = taken_from_mod.modulename.rpartition(".")[0]
            while identifier.startswith(".."):
                identifier = identifier[1:]
                parent_module = parent_module.rpartition(".")[0]
            identifier = parent_module + identifier
        else:
            # Check if this is a local reference within this module?
            for qualname in qualname_candidates(identifier, namespace):
                doc = mod.get(qualname)
                if doc and context["is_public"](doc).strip():
                    return f'<a href="#{qualname}">{plain_text}</a>'

        module = ""
        qualname = ""
        try:
            # Check if the object we are interested in is imported and re-exposed in the current namespace.
            for module, qualname in possible_sources(context["all_modules"], identifier):
                doc = mod.get(qualname)
                if doc and doc.taken_from == (module, qualname) and context["is_public"](doc).strip():
                    if plain_text.endswith("()"):
                        plain_text = f"{doc.qualname}()"
                    else:
                        plain_text = doc.qualname
                    return f'<a href="#{qualname}">{plain_text}</a>'
        except ValueError:
            # possible_sources did not find a parent module.
            return text
        else:
            # It's not, but we now know the parent module. Does the target exist?
            doc = context["all_modules"][module]
            if qualname:
                assert isinstance(doc, pdoc.doc.Module)
                try:
                    doc = doc.get(qualname)
                except Exception as ex:
                    print(f"Doc linkify issue: {qualname}:\n{ex}")
                    return text
            target_exists_and_public = doc is not None and context["is_public"](doc).strip()
            if target_exists_and_public:
                assert doc is not None  # mypy
                if qualname:
                    qualname = f"#{qualname}"
                if plain_text.endswith("()"):
                    plain_text = f"{doc.name}()"
                else:
                    plain_text = doc.name
                return f'<a href="{relative_link(context["module"].modulename, module)}{qualname}">{plain_text}</a>'
            else:
                return text

    return Markup(
        re.sub(
            r"""
            # Part 1: foo.bar or foo.bar() (without backticks)
            (?<![/=?#&])  # heuristic: not part of a URL
            # First part of the identifier (e.g. "foo") - this is optional for relative references.
            (?:
                \b
                (?!\d)[a-zA-Z0-9_]+
                |
                \.*  # We may also start with multiple dots.
            )
            # Rest of the identifier (e.g. ".bar" or "..bar")
            (?:
                # A single dot or a dot surrounded with pygments highlighting.
                (?:\.|</span><span\ class="o">\.</span><span\ class="n">)
                (?!\d)[a-zA-Z0-9_]+
            )+
            (?:\(\)|\b(?!\(\)))  # we either end on () or on a word boundary.
            (?!</a>)  # not an existing link
            (?![/#])  # heuristic: not part of a URL

            | # Part 2: `foo` or `foo()`. `foo.bar` is already covered with part 1.
            (?<=<code>)
                 (?!\d)[a-zA-Z0-9_]+
            (?:\(\))?
            (?=</code>(?!</a>))
            """,
            linkify_repl,
            code,
            flags=re.VERBOSE,
        )
    )


def generate_doc(
    package_name: str,
    package_version: str,
    package_url: Optional[str],
    output_path: Union[str, os.PathLike],
    package_paths: List[Union[str, os.PathLike]],
    optional_imports: Sequence[str] = (),
    extra_asset_path: Union[str, os.PathLike] = "doc",
    public_markdown_dir: Union[str, os.PathLike, None] = None,
    launch: bool = False,
):
    output_path = Path(output_path)
    extra_asset_path = Path(extra_asset_path)

    # clean doc dir
    if output_path.exists():
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

    # configure pdoc
    pdoc.render.configure(
        footer_text=f"{package_name} v{package_version}",
        logo_link=package_url,
        template_directory=extra_asset_path.joinpath("theme"),
        math=True,
        mermaid=True,
    )

    # add additional filters
    def remove_identifier(value: Markup):
        regex = r"(<a\s*href=\".*#\w*\">)(.*\.)([A-Z].*)(</a>)"
        subst = "\\1\\3\\4"
        result = re.sub(regex, subst, value, 0, re.MULTILINE)
        return Markup(result)

    pdoc.render.env.filters["remove_identifier"] = remove_identifier
    pdoc.render.env.filters["linkify"] = custom_linkify

    # add optional modules as patch
    pdoc_monkeypatch.patch()
    pdoc_monkeypatch.configure_top_level_markdown_override(
        package_name,
        Path(public_markdown_dir) if public_markdown_dir is not None else None,
    )
    stub_unavailable_imports(optional_imports)

    if launch:
        host = "localhost"
        port = 8052

        try:
            try:
                httpd = pdoc.web.DocServer((host, port or 8080), package_paths)
            except OSError:
                # Couldn't bind, let's try again with a random port.
                httpd = pdoc.web.DocServer((host, port or 0), package_paths)
        except OSError as e:
            print(f"Cannot start web server on {host}:{port}: {e}")
            sys.exit(1)

        with httpd:
            url = f"http://{host}:{httpd.server_port}"
            print(f"pdoc server ready at {url}")
            pdoc.web.open_browser(url)
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                httpd.server_close()
                return

    # generate pdoc
    pdoc.pdoc(*package_paths, output_directory=output_path)

    # copy doc content
    shutil.copytree(extra_asset_path, output_path.joinpath(extra_asset_path.name))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pdoc documentation for visiongraph.")
    parser.add_argument("--output", default="docs", help="Output path for the generated documentation.")
    parser.add_argument("--launch", action="store_true", help="Launch the pdoc web server instead of exiting.")
    parser.add_argument("--extra-assets", default="doc", help="Path to extra documentation assets.")
    parser.add_argument(
        "--public-markdown-dir",
        default=None,
        help="Optional directory containing build-time README.md and DOCUMENTATION.md overrides.",
    )
    args = parser.parse_args()

    from scripts.import_analyzer import VisiongraphAnalyzer

    config = load_project_config()
    result = VisiongraphAnalyzer().analyze()
    generate_doc(
        config.name,
        config.version,
        config.url,
        args.output,
        config.doc_modules,
        result.optional_imports,
        extra_asset_path=args.extra_assets,
        public_markdown_dir=args.public_markdown_dir,
        launch=args.launch,
    )


if __name__ == "__main__":
    main()
