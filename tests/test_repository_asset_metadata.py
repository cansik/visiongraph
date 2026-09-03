import ast
import unittest
from pathlib import Path

from scripts.model_asset_inventory import collect_asset_references


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "visiongraph"


def _is_repository_asset_call(node: ast.Call) -> bool:
    if isinstance(node.func, ast.Name):
        return node.func.id == "RepositoryAsset"

    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
        return node.func.value.id == "RepositoryAsset" and node.func.attr == "openVino"

    return False


def _iter_repository_asset_calls():
    for file_path in SOURCE_ROOT.rglob("*.py"):
        tree = ast.parse(file_path.read_text(), filename=str(file_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _is_repository_asset_call(node):
                yield file_path.relative_to(REPO_ROOT), node


class RepositoryAssetMetadataTests(unittest.TestCase):
    def test_corrected_model_licenses(self):
        from visiongraph.estimator.spatial.DEIMv2Detector import DEIMv2Config
        from visiongraph.estimator.spatial.face.emotion import FERPlusEmotionClassifier

        dino_asset, _labels = DEIMv2Config.DEIMv2_Dino3_S_COCO.value
        hgnet_asset, _labels = DEIMv2Config.DEIMv2_HgNetv2_Pico_COCO.value
        self.assertIn("DINOv3 License", dino_asset.metadata.license.name)
        self.assertEqual("Apache License 2.0", hgnet_asset.metadata.license.name)
        self.assertIn("3e491c6a6758f8f949e8dea7f5e4a28ab43334f5/LICENSE", hgnet_asset.metadata.license.url)
        self.assertIn("applied when these weights were released", dino_asset.metadata.comment)
        self.assertEqual(
            "https://github.com/microsoft/FERPlus/blob/master/LICENSE.md",
            FERPlusEmotionClassifier._MODEL_METADATA.license.url,
        )

    def test_all_discovered_repository_assets_have_complete_metadata(self):
        references, _failed_imports = collect_asset_references(repository_only=True)
        missing_metadata = []

        for reference in references:
            metadata = reference.asset.metadata
            if metadata is None:
                missing_metadata.append(
                    f"{reference.module_name}.{reference.enum_name}.{reference.member_name}: {reference.asset_name}"
                )
                continue

            if not metadata.source.name or not metadata.source.url:
                missing_metadata.append(
                    f"{reference.module_name}.{reference.enum_name}.{reference.member_name}: {reference.asset_name}"
                    " has incomplete source metadata"
                )

            if not metadata.license.name or not metadata.license.url:
                missing_metadata.append(
                    f"{reference.module_name}.{reference.enum_name}.{reference.member_name}: {reference.asset_name}"
                    " has incomplete license metadata"
                )

        self.assertEqual([], missing_metadata, "Repository assets missing metadata:\n" + "\n".join(missing_metadata))

    def test_all_repository_asset_calls_pass_metadata_keyword(self):
        missing_metadata_keyword = []

        for file_path, node in _iter_repository_asset_calls():
            if any(keyword.arg == "metadata" for keyword in node.keywords):
                continue

            missing_metadata_keyword.append(f"{file_path}:{node.lineno}")

        self.assertEqual(
            [],
            missing_metadata_keyword,
            "RepositoryAsset calls without metadata keyword:\n" + "\n".join(missing_metadata_keyword),
        )


if __name__ == "__main__":
    unittest.main()
