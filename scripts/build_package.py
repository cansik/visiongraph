import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from scripts import generate_public_markdown


REPO_ROOT = Path(__file__).resolve().parents[1]
COPY_IGNORE = shutil.ignore_patterns(
    ".git",
    ".idea",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "docs",
    "*.egg-info",
)


def stage_repository_tree(repo_root: Path, staging_root: Path) -> None:
    shutil.copytree(repo_root, staging_root, ignore=COPY_IGNORE)
    generate_public_markdown.generate_public_markdown(output_dir=staging_root, repo_root=staging_root)


def build_package(repo_root: Path = REPO_ROOT, dist_dir: Path | None = None, uv_binary: str = "uv") -> None:
    repo_root = repo_root.resolve()
    if dist_dir is None:
        dist_dir = repo_root / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="visiongraph-build-") as temp_dir:
        staging_root = Path(temp_dir) / repo_root.name
        stage_repository_tree(repo_root, staging_root)
        subprocess.run([uv_binary, "build", "--out-dir", str(dist_dir)], cwd=staging_root, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build wheel and sdist artifacts from a staged repository tree.")
    parser.add_argument(
        "--dist-dir", type=Path, default=REPO_ROOT / "dist", help="Output directory for build artifacts."
    )
    parser.add_argument("--uv-binary", default="uv", help="UV executable used to build the package.")
    args = parser.parse_args()

    build_package(dist_dir=args.dist_dir, uv_binary=args.uv_binary)


if __name__ == "__main__":
    main()
