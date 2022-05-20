import argparse
from pathlib import Path

from tqdm import tqdm

from tools import utils


def main():
    bins = utils.get_files_in_path(args.path, ["**/*.bin"])
    with tqdm(total=len(bins), desc="prepare") as pbar:
        for path in bins:
            bin = Path(path)
            xml = Path(path.replace(".bin", ".xml"))
            format = bin.parent.name.lower()

            if not xml.exists():
                pbar.update()
                continue

            if not bin.name.endswith(f"-{format}.bin"):
                bin.replace(str(bin).replace(".bin", f"-{format}.bin"))

            if not xml.name.endswith(f"-{format}.xml"):
                xml.replace(str(xml).replace(".xml", f"-{format}.xml"))
            pbar.update()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Model Renamer", description="Renames the models to be ready for upload.")
    parser.add_argument("path", type=str, help="Path to the root folder of the models.")
    args = parser.parse_args()

    main()
