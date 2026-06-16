# AGENTS.md

This file is for coding agents and automated contributors working in this repository. Follow it as the default contribution guide for repository changes.

## Repository Overview

- `visiongraph/` contains the runtime library.
- `visiongraph/vg/__init__.py` is generated and provides the lazy public API surface used through `from visiongraph import vg`.
- `scripts/` contains repository-maintenance and generation utilities such as init generation, docs generation, estimator listing, model attribution generation, license checks, and inventory helpers.
- `tests/` contains the unittest-based test suite.
- `README.md` contains install, build, and workflow guidance.
- `DOCUMENTATION.md` contains conceptual and usage documentation.
- `MODEL_ATTRIBUTIONS.md` is generated from repository asset metadata.
- `pyproject.toml` is the source of truth for packaging, build backend, dependencies, and Hatchling file inclusion.

## Required After Code Changes

After each repository code change, run these steps before considering the work finished:

1. `make generate-artifacts`
2. `make autoformat`
3. Run relevant tests for the touched area
4. Update `README.md` and/or `DOCUMENTATION.md` when public behavior, public API, build workflow, extras, packaging, or usage changes

`make generate-artifacts` is the aggregation point for generated repository artifacts required for distribution. At the moment it regenerates:

- `visiongraph/vg/__init__.py`
- `MODEL_ATTRIBUTIONS.md`

If a future change introduces another generated artifact needed for packaging or distribution, extend `make generate-artifacts` and make CI use that target before build.

## Build And Distribution Rules

- The build backend is Hatchling.
- Define wheel and sdist file inclusion in `pyproject.toml`, not in `MANIFEST.in`.
- `make build` is the standard path for creating the sdist and wheel.
- For build-related CI or release steps, generate repository artifacts first by using `make generate-artifacts` before `uv build`.

## Scripts Vs Library Code

- Put runtime library code in `visiongraph/`.
- Put repository-maintenance utilities in `scripts/`.
- `scripts/` is the correct place for tasks like inventory generation, attribution generation, estimator listing, license checks, documentation generation, and similar repo tooling.
- Avoid adding new repo-maintenance utilities to `tools/` when they are really scripts used by maintainers, CI, or packaging flows.

## Documentation Standards

### Code Docstrings

- Follow the existing repository style for public classes, methods, properties, and other public APIs.
- Use triple-double-quoted docstrings.
- Start with a short summary sentence.
- Add `:param ...:` and `:return:` sections when they improve clarity.
- Keep docstrings descriptive and behavioral. Do not restate obvious code line by line.
- Private helpers do not need docstrings unless the behavior is non-obvious or the logic is easy to misuse.

Representative style examples live in files such as:

- `visiongraph.GraphNode.GraphNode`
- `visiongraph.input.BaseDepthCamera.BaseDepthCamera.get_image()`

### Markdown Documentation

- Use `README.md` for install, build, packaging, and quick-start workflow guidance.
- Use `DOCUMENTATION.md` for conceptual explanations, architecture, and usage patterns.
- When referring to repository symbols in markdown docs, prefer backticked fully qualified identifiers instead of hardcoded links.
- These identifiers are resolved into links later by the documentation renderer.

Valid examples:

- `visiongraph.GraphNode.GraphNode`
- `visiongraph.input.BaseDepthCamera.BaseDepthCamera.get_image()`

- Prefer these identifiers over manual markdown links for API references inside repo documentation.

## Code Style Expectations

- Prefer small, focused modules over large grab-bag files.
- Use modern Python style and modern type hints.
- Prefer built-in generic syntax like `list[str]`, `dict[str, int]`, and union syntax like `str | None`.
- Use explicit type annotations on public APIs.
- Keep naming descriptive and avoid one-letter variables except for very local conventional cases.
- Prefer `pathlib.Path`, dataclasses, enums, and clear structured types where they improve clarity.
- Keep imports explicit.
- Localize expensive or optional imports when startup behavior, optional dependencies, or platform availability matters.
- Match existing repository conventions before introducing a different pattern.

Even when using modern style, keep code compatible with the Python versions declared in `pyproject.toml`.

## Testing Guidance

- Run the narrowest relevant tests first.
- Expand validation only as needed for the changed surface.
- Use these common validation commands when applicable:

- `make lint`
- `make test-metadata`
- `make list-estimators`
- `make docs`
- `make build`

- For repository asset metadata or repository-backed model config changes, always run `make test-metadata`.
- For packaging or generated-artifact changes, run `make generate-artifacts` and `make build`.
- For documentation renderer or docs-generation changes, run `make docs`.

## Asset Metadata Rules

- Repository-backed assets must carry complete metadata.
- Define metadata in the owning model module by default.
- Use shared metadata only when the same metadata is reused across multiple modules.
- Shared metadata constants should end in `_METADATA`.
- Regenerate `MODEL_ATTRIBUTIONS.md` after relevant metadata or asset inventory changes.

## Agent Behavior Expectations

- Do not hand-edit generated files and leave them stale; regenerate them.
- Do not introduce build or CI steps that bypass the Makefile workflow without a reason.
- When changing public APIs, examples, commands, or packaging behavior, update the docs in the same change.
- Keep the repository internally consistent rather than adding one-off exceptions.