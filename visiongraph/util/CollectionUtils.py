from collections import defaultdict
from typing import Any, Dict


def default_value_dict(value: Any, source: Dict) -> Dict:
    d = defaultdict(lambda: value)
    for k, v in source.items():
        d[k] = v
    return d
