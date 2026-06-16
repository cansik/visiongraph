import argparse
import re
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "build" / "public-markdown"
SOURCE_MARKDOWN_FILES = ("README.md", "DOCUMENTATION.md")
MARKDOWN_LINK_RE = re.compile(r"(!?)\[([^\]]+)\]\(([^)]+)\)")


def load_repository_url(pyproject_path: Path = REPO_ROOT / "pyproject.toml") -> str:
    if pyproject_path.suffix != ".toml":
        raise ValueError(f"Expected a pyproject.toml file, got: {pyproject_path}")

    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
        import tomli as tomllib

    with pyproject_path.open("rb") as file:
        data = tomllib.load(file)

    urls = data["project"].get("urls", {})
    repository_url = urls.get("Repository") or urls.get("Homepage")
    if not repository_url:
        raise ValueError("Could not determine repository URL from pyproject.toml")

    return repository_url.removesuffix(".git").rstrip("/")


def is_relative_target(target: str) -> bool:
    return not (
        "://" in target
        or target.startswith(("#", "mailto:", "tel:"))
        or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target) is not None
    )


def build_public_url(repository_url: str, ref: str, relative_path: Path, is_image: bool, is_directory: bool) -> str:
    path_text = relative_path.as_posix()
    if is_image:
        return f"{repository_url}/raw/{ref}/{path_text}"

    target_kind = "tree" if is_directory else "blob"
    return f"{repository_url}/{target_kind}/{ref}/{path_text}"


def rewrite_markdown_links(
    text: str,
    source_path: Path,
    repository_url: str,
    ref: str = "main",
    repo_root: Path | None = None,
) -> str:
    if repo_root is None:
        repo_root = REPO_ROOT
    repo_root = repo_root.resolve()
    source_path = source_path.resolve()

    def replace(match: re.Match[str]) -> str:
        is_image = bool(match.group(1))
        label = match.group(2)
        target = match.group(3)

        if not is_relative_target(target):
            return match.group(0)

        if "#" in target:
            raw_target, anchor = target.split("#", 1)
            anchor_suffix = f"#{anchor}"
        else:
            raw_target = target
            anchor_suffix = ""

        resolved_target = (source_path.parent / raw_target).resolve()

        try:
            relative_path = resolved_target.relative_to(repo_root)
        except ValueError:
            return match.group(0)

        if not resolved_target.exists():
            return match.group(0)

        public_url = build_public_url(
            repository_url, ref, relative_path, is_image=is_image, is_directory=resolved_target.is_dir()
        )
        return f"{'!' if is_image else ''}[{label}]({public_url}{anchor_suffix})"

    return MARKDOWN_LINK_RE.sub(replace, text)


def generate_public_markdown(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    ref: str = "main",
    repo_root: Path | None = None,
    source_names: Sequence[str] = SOURCE_MARKDOWN_FILES,
) -> list[Path]:
    if repo_root is None:
        repo_root = REPO_ROOT
    repo_root = repo_root.resolve()
    repository_url = load_repository_url(repo_root / "pyproject.toml")
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_paths: list[Path] = []
    for source_name in source_names:
        source_path = repo_root / source_name
        rewritten = rewrite_markdown_links(
            source_path.read_text(encoding="utf-8"),
            source_path,
            repository_url,
            ref=ref,
            repo_root=repo_root,
        )
        target_path = output_dir / source_name
        target_path.write_text(rewritten, encoding="utf-8")
        generated_paths.append(target_path)

    return generated_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate public markdown files with absolute GitHub links.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--ref", default="main", help="Git reference used in generated public links.")
    args = parser.parse_args()

    generate_public_markdown(output_dir=args.output_dir, ref=args.ref)


if __name__ == "__main__":
    main()
