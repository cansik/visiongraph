import argparse
import re
from enum import Enum
from pathlib import Path
from typing import List

import tqdm


class SectionType(Enum):
    Parameter = 0
    Return = 1
    Raise = 2
    Null = 5


def convert_docstring_to_rst(code: str) -> str:
    """
    Converts Python docstrings in the code to reStructuredText format.

    :param code: The Python code with docstrings to be converted.
    :return: Python code with reStructuredText formatted docstrings.
    """

    params_regex = r"^\s*([\*\w_]+)\s*(\(([\w\s\_\[\],\.]+)\))?\s*:\s*(.*)$"
    return_regex = r"^\s*([\w\s\_\[\],\.]*)\s*:\s*(.*)\s*$"
    raise_regex = r"^\s*([\w\s\_\[\],\.]*)\s*:\s*(.*)\s*$"

    def extract_and_convert_docstring(match: re.Match) -> str:
        docstring = match.group(1)
        rst_lines = []

        section = SectionType.Null
        spaces_count = 0
        intent = ""

        for line in docstring.splitlines():
            spaces_count = len(line) - len(line.lstrip(" "))
            line = line.strip()
            if line.startswith("Args:"):
                intent = " " * spaces_count
                section = SectionType.Parameter
            elif line.startswith("Returns:"):
                intent = " " * spaces_count
                section = SectionType.Return
            elif line.startswith("Raises:"):
                intent = " " * spaces_count
                section = SectionType.Raise
            else:
                if section == SectionType.Parameter and line:
                    matches = list(re.finditer(params_regex, line, re.MULTILINE))
                    if len(matches) == 0:
                        continue
                    m = matches[0]

                    variable_name = m.group(1)
                    type_def = m.group(2)
                    description = m.group(4)

                    # todo: extract the correct line infos
                    rst_lines.append(f"{intent}:param {variable_name}: {description.strip()}")
                elif section == SectionType.Return and line:
                    matches = list(re.finditer(return_regex, line, re.MULTILINE))

                    if len(matches) == 0:
                        continue
                    m = matches[0]

                    type_def = m.group(1)
                    description = m.group(2)

                    rst_lines.append(f"{intent}:return: {description.strip()}")
                elif section == SectionType.Raise and line:
                    matches = list(re.finditer(raise_regex, line, re.MULTILINE))

                    if len(matches) == 0:
                        continue
                    m = matches[0]

                    type_def = m.group(1)
                    description = m.group(2)

                    rst_lines.append(f"{intent}:raises {type_def}: {description.strip()}")
                else:
                    intent = " " * spaces_count
                    rst_lines.append(f"{intent}{line}")

        if len(rst_lines) > 0:
            if rst_lines[0].strip() == "":
                rst_lines.pop(0)

            if rst_lines[-1].strip() == "":
                rst_lines.pop(-1)

        return '"""' + "\n" + "\n".join(rst_lines) + "\n" + intent + '"""'

    # Replace Python docstrings with reStructuredText docstrings
    rst_code = re.sub(
        r'"""(\s*.*?\s*)"""',
        extract_and_convert_docstring,
        code,
        flags=re.DOTALL
    )

    return rst_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", default=".", help="Directory to find python files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    files: List[Path] = list(Path(args.directory).rglob("*.py"))

    for file in tqdm.tqdm(files, "converting"):
        text = file.read_text(encoding="utf-8")

        # extract comment sections
        corrected = convert_docstring_to_rst(text)
        file.write_text(corrected, encoding="utf-8")
        # print(corrected)
        # exit(0)

    print("done!")


if __name__ == "__main__":
    main()
