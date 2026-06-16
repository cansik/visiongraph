.PHONY: \
	autoformat \
	attributions \
	build \
	build-ci \
	check \
	ci-code \
	clean \
	clean-dist \
	clean-docs \
	default \
	download-model-licenses \
	docs \
	docs-serve \
	fmt \
	fmt-check \
	generate-init \
	generate-public-markdown \
	generate-artifacts \
	help \
	license-check \
	lint \
	list-estimators \
	sync \
	test \
	test-metadata

UV := uv
PYTHON := $(UV) run python
ATTRIBUTIONS_FILE := MODEL_ATTRIBUTIONS.md

default: autoformat

help:
	@echo "Available targets:"
	@echo "  sync              Install project dependencies used by local development, docs, and CI"
	@echo "  fmt               Format the repository with Ruff"
	@echo "  fmt-check         Check formatting with Ruff"
	@echo "  lint              Run Ruff lint checks"
	@echo "  autoformat        Format the repository and apply auto-fixable Ruff rules"
	@echo "  generate-init     Regenerate the top-level visiongraph __init__.py"
	@echo "  generate-public-markdown Generate build-only markdown variants with absolute GitHub links"
	@echo "  attributions      Generate $(ATTRIBUTIONS_FILE) from repository model metadata"
	@echo "  download-model-licenses Download model license files into build/licenses"
	@echo "  generate-artifacts Regenerate repository-generated artifacts committed to the repository"
	@echo "  list-estimators   Print discovered estimator config enums"
	@echo "  license-check     Print dependency licenses from pyproject.toml"
	@echo "  test              Run the full unittest suite"
	@echo "  test-metadata     Run the repository model metadata enforcement tests"
	@echo "  docs              Generate the static documentation into ./docs"
	@echo "  docs-serve        Launch the pdoc documentation web server"
	@echo "  build             Regenerate distributable artifacts and build sdist/wheel artifacts"
	@echo "  build-ci          Alias for the CI package build sequence"
	@echo "  clean-docs        Remove generated documentation"
	@echo "  clean-dist        Remove generated build artifacts"
	@echo "  clean             Remove generated documentation and build artifacts"

sync:
	$(UV) sync --all-extras --dev --group docs

fmt:
	$(UV) run ruff format .

fmt-check:
	$(UV) run ruff format --check .

lint:
	$(UV) run ruff check .

autoformat:
	$(UV) run ruff format . && $(UV) run ruff check --fix .

ci-code: lint fmt-check

check: ci-code test

generate-init:
	$(PYTHON) -m scripts.generate_init

generate-public-markdown:
	$(PYTHON) -m scripts.generate_public_markdown

attributions:
	$(PYTHON) -m scripts.generate_model_attributions --output $(ATTRIBUTIONS_FILE)

download-model-licenses:
	$(PYTHON) -m scripts.download_model_licenses --output-dir build/licenses

generate-artifacts: generate-init attributions

list-estimators:
	$(PYTHON) -m scripts.list_estimators

license-check:
	$(PYTHON) -m scripts.license_check

test:
	$(PYTHON) -m unittest discover -s ./tests -t ./ -v

test-metadata:
	$(PYTHON) -m unittest tests.test_repository_asset_metadata -v

docs: generate-init generate-public-markdown
	$(PYTHON) -m scripts.generate_doc --public-markdown-dir build/public-markdown

docs-serve: generate-init
	$(PYTHON) -m scripts.generate_doc --launch

build: generate-artifacts
	$(PYTHON) -m scripts.build_package

build-ci: build

clean-docs:
	rm -rf docs

clean-dist:
	rm -rf build dist *.egg-info

clean: clean-docs clean-dist