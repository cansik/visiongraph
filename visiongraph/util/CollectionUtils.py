from collections import defaultdict
from typing import Any, Dict


def default_value_dict(value: Any, source: Dict) -> Dict:
    """
    Creates a dictionary with a default value for keys that do not exist.

    Args:
        value (Any): The default value to use for keys not found in source.
        source (Dict): A dictionary from which to create the new dictionary.

    Returns:
        Dict: A dictionary where each key in source maps to its value,
              and missing keys map to the default value.
    """
    d = defaultdict(lambda: value)
    for k, v in source.items():
        d[k] = v
    return d
