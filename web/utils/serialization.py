"""Serialization helpers for API responses and database JSON fields."""

import math
from datetime import date, datetime
from enum import Enum
from typing import Any


def to_json_compatible(value: Any) -> Any:
    """Recursively convert common scientific Python values to strict JSON.

    Non-finite numbers become ``None``. NumPy and pandas are optional, so the
    helper keeps those imports local and degrades safely when they are absent.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, Enum):
        return to_json_compatible(value.value)

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, dict):
        return {
            str(to_json_compatible(key)): to_json_compatible(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_json_compatible(item) for item in value]

    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return to_json_compatible(value.tolist())
        if isinstance(value, np.generic):
            return to_json_compatible(value.item())
    except ImportError:
        pass

    try:
        import pandas as pd

        is_missing = pd.isna(value)
        if not hasattr(is_missing, "__len__") and bool(is_missing):
            return None
    except (ImportError, TypeError, ValueError):
        pass

    return str(value)
