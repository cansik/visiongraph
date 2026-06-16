import contextlib
import enum
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import (
    build_package,
    generate_model_attributions,
    generate_public_markdown,
    import_analyzer,
    license_check,
    list_estimators,
    model_asset_inventory,
    pdoc_monkeypatch,
)
from scripts.model_asset_inventory import AssetReference, ConfigEnumReference, FailedImport
from visiongraph.data.AssetMetadata import AssetMetadata
from visiongraph.data.LocalAsset import LocalAsset
from visiongraph.data.RepositoryAsset import RepositoryAsset


class ModelAssetInventoryTests(unittest.TestCase):
    def test_iter_config_enums_collects_unique_enum_configs_and_failed_imports(self):
        class GoodConfig(enum.Enum):
            FIRST = 1

        class NotAConfig:
            pass

        class _LazyValue:
            def __init__(self, module_name, value=None, error=None):
                self.module_name = module_name
                self._value = value
                self._error = error

            @property
            def attribute(self):
                if self._error is not None:
                    raise self._error
                return self._value

        fake_imports = {
            "GoodConfig": _LazyValue("visiongraph.fake.good", GoodConfig),
            "DuplicateConfig": _LazyValue("visiongraph.fake.duplicate", GoodConfig),
            "BrokenConfig": _LazyValue("visiongraph.fake.broken", error=RuntimeError("boom")),
            "OtherThing": _LazyValue("visiongraph.fake.other", NotAConfig),
        }

        with mock.patch.object(model_asset_inventory.vg, "_visiongraph_imports", fake_imports):
            config_enums, failed_imports = model_asset_inventory.iter_config_enums()

        self.assertEqual(1, len(config_enums))
        self.assertEqual("GoodConfig", config_enums[0].enum_name)
        self.assertIs(GoodConfig, config_enums[0].enum_class)
        self.assertEqual(1, len(failed_imports))
        self.assertEqual("BrokenConfig", failed_imports[0].import_name)
        self.assertEqual("visiongraph.fake.broken", failed_imports[0].module_name)
        self.assertIn("boom", failed_imports[0].error)

    def test_collect_asset_references_filters_repository_assets_and_deduplicates_nested_values(self):
        metadata = AssetMetadata.from_values(
            "Source", "https://example.com/source", "MIT", "https://example.com/license"
        )
        repo_asset = RepositoryAsset("model.onnx", metadata=metadata)
        local_asset = LocalAsset("local.file")

        class FakeConfig(enum.Enum):
            ENTRY = (repo_asset, {"duplicate": repo_asset, "local": local_asset})

        config_refs = [ConfigEnumReference("visiongraph.fake.module", "FakeConfig", FakeConfig)]
        failed_imports = [FailedImport("BrokenConfig", "visiongraph.fake.module", "boom")]

        with mock.patch("scripts.model_asset_inventory.iter_config_enums", return_value=(config_refs, failed_imports)):
            references, discovered_failures = model_asset_inventory.collect_asset_references(repository_only=True)

        self.assertEqual(failed_imports, list(discovered_failures))
        self.assertEqual(1, len(references))
        reference = references[0]
        self.assertEqual("visiongraph.fake.module", reference.module_name)
        self.assertEqual("FakeConfig", reference.enum_name)
        self.assertEqual("ENTRY", reference.member_name)
        self.assertEqual("model.onnx", reference.asset_name)
        self.assertIs(repo_asset, reference.asset)


class GenerateModelAttributionsTests(unittest.TestCase):
    def test_build_markdown_groups_assets_and_reports_missing_and_skipped_entries(self):
        metadata = AssetMetadata.from_values(
            "Grouped Source",
            "https://example.com/source",
            "Apache License 2.0",
            "https://example.com/license",
        )
        grouped_repo_a = RepositoryAsset("b-model.onnx", metadata=metadata)
        grouped_repo_b = RepositoryAsset("a-model.onnx", metadata=metadata)
        missing_repo = RepositoryAsset("missing.onnx")

        references = [
            AssetReference("visiongraph.fake", "Cfg", "B", grouped_repo_a.name, grouped_repo_a),
            AssetReference("visiongraph.fake", "Cfg", "A", grouped_repo_b.name, grouped_repo_b),
            AssetReference("visiongraph.fake", "Cfg", "MISSING", missing_repo.name, missing_repo),
        ]
        failed_imports = [FailedImport("BrokenConfig", "visiongraph.fake", "boom")]

        with mock.patch(
            "scripts.generate_model_attributions.collect_asset_references",
            return_value=(references, failed_imports),
        ):
            markdown = generate_model_attributions.build_markdown()

        self.assertIn("# Model Attributions", markdown)
        self.assertIn("## Grouped Source", markdown)
        self.assertIn("Origin: https://example.com/source", markdown)
        self.assertIn("License: [Apache License 2.0](https://example.com/license)", markdown)
        self.assertLess(markdown.index("- a-model.onnx"), markdown.index("- b-model.onnx"))
        self.assertIn("## Missing Metadata", markdown)
        self.assertIn("missing.onnx (visiongraph.fake.Cfg.MISSING)", markdown)
        self.assertIn("## Skipped Imports", markdown)
        self.assertIn("BrokenConfig (visiongraph.fake): boom", markdown)


class GeneratePublicMarkdownTests(unittest.TestCase):
    def test_rewrite_markdown_links_converts_repo_relative_targets_to_absolute_github_urls(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            source_path = temp_root / "README.md"
            (temp_root / "examples").mkdir()
            (temp_root / "examples" / "Demo.py").write_text("print('demo')\n", encoding="utf-8")
            (temp_root / "doc").mkdir()
            (temp_root / "doc" / "sample.webp").write_bytes(b"img")
            (temp_root / "LICENSE").write_text("MIT\n", encoding="utf-8")

            source_path.write_text(
                "\n".join(
                    [
                        "[Examples](examples)",
                        "[Demo](examples/Demo.py)",
                        "![Sample](doc/sample.webp)",
                        "[License](LICENSE)",
                        "[Anchor](#local)",
                        "[External](https://example.com)",
                    ]
                ),
                encoding="utf-8",
            )

            with mock.patch.object(generate_public_markdown, "REPO_ROOT", temp_root):
                rewritten = generate_public_markdown.rewrite_markdown_links(
                    source_path.read_text(encoding="utf-8"),
                    source_path,
                    "https://github.com/example/project",
                    ref="main",
                )

        self.assertIn("[Examples](https://github.com/example/project/tree/main/examples)", rewritten)
        self.assertIn("[Demo](https://github.com/example/project/blob/main/examples/Demo.py)", rewritten)
        self.assertIn("![Sample](https://github.com/example/project/raw/main/doc/sample.webp)", rewritten)
        self.assertIn("[License](https://github.com/example/project/blob/main/LICENSE)", rewritten)
        self.assertIn("[Anchor](#local)", rewritten)
        self.assertIn("[External](https://example.com)", rewritten)

    def test_generate_public_markdown_writes_rewritten_root_markdown_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            output_dir = temp_root / "build" / "public-markdown"
            (temp_root / "examples").mkdir()
            (temp_root / "examples" / "Demo.py").write_text("print('demo')\n", encoding="utf-8")
            (temp_root / "pyproject.toml").write_text(
                "\n".join(
                    [
                        "[project]",
                        'name = "visiongraph"',
                        'version = "0.0.0"',
                        'readme = "README.md"',
                        "[project.urls]",
                        'Repository = "https://github.com/example/project.git"',
                    ]
                ),
                encoding="utf-8",
            )
            (temp_root / "README.md").write_text("[Demo](examples/Demo.py)\n", encoding="utf-8")
            (temp_root / "DOCUMENTATION.md").write_text("[Docs](README.md)\n", encoding="utf-8")

            generated_paths = generate_public_markdown.generate_public_markdown(
                output_dir=output_dir, repo_root=temp_root
            )

            self.assertEqual([output_dir / "README.md", output_dir / "DOCUMENTATION.md"], generated_paths)
            generated_readme = generated_paths[0].read_text(encoding="utf-8")

        self.assertIn(
            "[Demo](https://github.com/example/project/blob/main/examples/Demo.py)",
            generated_readme,
        )


class PdocMonkeypatchTests(unittest.TestCase):
    def test_build_top_level_docstring_points_to_absolute_markdown_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "README.md").write_text("README\n", encoding="utf-8")
            (temp_path / "DOCUMENTATION.md").write_text("DOCS\n", encoding="utf-8")

            docstring = pdoc_monkeypatch.build_top_level_docstring(temp_path)

        self.assertIn(str((temp_path / "README.md").resolve()), docstring)
        self.assertIn(str((temp_path / "DOCUMENTATION.md").resolve()), docstring)


class ImportAnalyzerTests(unittest.TestCase):
    def test_analyze_propagates_optional_modules_through_excluded_relative_imports(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            package_root = temp_path / "samplepkg"
            external_root = package_root / "external" / "optional_pkg"
            external_root.mkdir(parents=True)

            (package_root / "__init__.py").write_text("\n", encoding="utf-8")
            (package_root / "consumer.py").write_text(
                "from samplepkg.external.optional_pkg import OptionalThing\n",
                encoding="utf-8",
            )
            (package_root / "external" / "__init__.py").write_text("\n", encoding="utf-8")
            (external_root / "__init__.py").write_text(
                "from .module import OptionalThing\n",
                encoding="utf-8",
            )
            (external_root / "module.py").write_text(
                "from filterpy.common import helper\n\nclass OptionalThing:\n    pass\n",
                encoding="utf-8",
            )

            analyzer = import_analyzer.ModuleAnalyzer(
                root_package="samplepkg",
                excluded_modules={"samplepkg.external"},
                late_import_modules=set(),
                optional_modules={"filterpy"},
                module_with_methods=set(),
            )

            with contextlib.chdir(temp_path):
                result = analyzer.analyze()

        self.assertIn("samplepkg.consumer", result.optional_modules)
        self.assertIn("samplepkg.external.optional_pkg.__init__", result.optional_modules)
        self.assertIn("samplepkg.external.optional_pkg.module", result.optional_modules)


class BuildPackageTests(unittest.TestCase):
    def test_stage_repository_tree_overlays_public_markdown_at_repo_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            repo_root = temp_root / "repo"
            staging_root = temp_root / "staged"
            repo_root.mkdir()
            (repo_root / "examples").mkdir()
            (repo_root / "examples" / "Demo.py").write_text("print('demo')\n", encoding="utf-8")
            (repo_root / "pyproject.toml").write_text(
                "\n".join(
                    [
                        "[project]",
                        'name = "visiongraph"',
                        'version = "0.0.0"',
                        'readme = "README.md"',
                        "[project.urls]",
                        'Repository = "https://github.com/example/project.git"',
                    ]
                ),
                encoding="utf-8",
            )
            (repo_root / "README.md").write_text("[Demo](examples/Demo.py)\n", encoding="utf-8")
            (repo_root / "DOCUMENTATION.md").write_text("Documentation\n", encoding="utf-8")

            build_package.stage_repository_tree(repo_root, staging_root)

            staged_readme = (staging_root / "README.md").read_text(encoding="utf-8")

        self.assertIn("[Demo](https://github.com/example/project/blob/main/examples/Demo.py)", staged_readme)


class ListEstimatorsTests(unittest.TestCase):
    def test_main_prints_modules_members_and_failed_import_warnings(self):
        class AlphaConfig(enum.Enum):
            FIRST = 1
            SECOND = 2

        class BetaConfig(enum.Enum):
            ENTRY = 1

        config_enums = [
            ConfigEnumReference("visiongraph.alpha", "AlphaConfig", AlphaConfig),
            ConfigEnumReference("visiongraph.beta", "BetaConfig", BetaConfig),
        ]
        failed_imports = [FailedImport("BrokenConfig", "visiongraph.beta", "boom")]

        output = io.StringIO()
        with mock.patch("scripts.model_asset_inventory.iter_config_enums", return_value=(config_enums, failed_imports)):
            with contextlib.redirect_stdout(output):
                list_estimators.main()

        text = output.getvalue()
        self.assertIn("- `visiongraph.alpha` ()", text)
        self.assertIn(" - FIRST", text)
        self.assertIn(" - SECOND", text)
        self.assertIn("- `visiongraph.beta` ()", text)
        self.assertIn(" - ENTRY", text)
        self.assertIn("[warn] could not import BrokenConfig: boom", text)


class LicenseCheckTests(unittest.TestCase):
    def test_parse_pyproject_collects_project_optional_and_dependency_group_requirements(self):
        pyproject_text = """
[project]
dependencies = ["requests>=2", "opencv-python~=4.12"]

[project.optional-dependencies]
media = ["moviepy", "vidgear[core]~=0.3.3"]

[dependency-groups]
dev = ["pytest", "ruff>=0.12"]
"""

        with tempfile.TemporaryDirectory() as temp_dir:
            pyproject_path = Path(temp_dir) / "pyproject.toml"
            pyproject_path.write_text(pyproject_text, encoding="utf-8")

            requirements = license_check.parse_pyproject(str(pyproject_path))

        self.assertEqual(
            {"requests", "opencv-python", "moviepy", "vidgear", "pytest", "ruff"},
            requirements,
        )

    def test_normalize_license_maps_common_values_and_unknowns(self):
        self.assertEqual("MIT License", license_check.normalize_license("MIT"))
        self.assertEqual("Apache License 2.0", license_check.normalize_license("Apache License, Version 2.0"))
        self.assertEqual("Unknown", license_check.normalize_license("Completely Custom License"))


if __name__ == "__main__":
    unittest.main()
