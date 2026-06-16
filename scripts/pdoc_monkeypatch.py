import warnings
from pathlib import Path

import pdoc
from pdoc import render, doc, extract


TOP_LEVEL_MARKDOWN_FILES = ("README.md", "DOCUMENTATION.md")
_top_level_markdown_overrides: dict[str, Path] = {}


def build_top_level_docstring(markdown_dir: Path) -> str:
    return "\n".join(f".. include:: {(markdown_dir / name).resolve()}" for name in TOP_LEVEL_MARKDOWN_FILES)


def configure_top_level_markdown_override(module_name: str, markdown_dir: Path | None) -> None:
    if markdown_dir is None:
        _top_level_markdown_overrides.pop(module_name, None)
        return

    _top_level_markdown_overrides[module_name] = markdown_dir.resolve()


def _apply_top_level_markdown_overrides(all_modules: dict[str, doc.Module]) -> None:
    for module_name, markdown_dir in _top_level_markdown_overrides.items():
        module = all_modules.get(module_name)
        if module is None:
            continue

        missing_files = [name for name in TOP_LEVEL_MARKDOWN_FILES if not markdown_dir.joinpath(name).exists()]
        if missing_files:
            warnings.warn(
                f"Skipping markdown override for {module_name}: missing files in {markdown_dir}: {', '.join(missing_files)}"
            )
            continue

        module.docstring = build_top_level_docstring(markdown_dir)


def patched_pdoc(
    *modules: Path | str,
    output_directory: Path | None = None,
) -> str | None:
    """
    Render the documentation for a list of modules.

     - If `output_directory` is `None`, returns the rendered documentation
       for the first module in the list.
     - If `output_directory` is set, recursively writes the rendered output
       for all specified modules and their submodules to the target destination.

    Rendering options can be configured by calling `pdoc.render.configure` in advance.
    """
    doc.Module.from_name.cache_clear()
    pdoc.docstrings.convert.cache_clear()

    all_modules: dict[str, doc.Module] = {}
    for module_name in extract.walk_specs(modules):
        all_modules[module_name] = doc.Module.from_name(module_name)

    # filter AutoMock objects
    all_modules = {k: v for k, v in all_modules.items() if isinstance(v.modulename, str)}
    _apply_top_level_markdown_overrides(all_modules)

    for module in all_modules.values():
        out = render.html_module(module, all_modules)
        if not output_directory:
            return out
        else:
            outfile = output_directory / f"{module.fullname.replace('.', '/')}.html"
            outfile.parent.mkdir(parents=True, exist_ok=True)
            outfile.write_bytes(out.encode())

    assert output_directory

    index = render.html_index(all_modules)
    if index:
        (output_directory / "index.html").write_bytes(index.encode())

    search = render.search_index(all_modules)
    if search:
        (output_directory / "search.js").write_bytes(search.encode())

    return None


def patch():
    pdoc.pdoc = patched_pdoc
