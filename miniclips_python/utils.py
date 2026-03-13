"""Small helper functions shared across modules."""

from __future__ import annotations

from typing import Dict

import numpy as np


def goal_prob_dict(goal_names: list[str], probs: np.ndarray) -> Dict[str, float]:
    """Convenience formatter: {goal_name: probability}."""

    return {g: float(p) for g, p in zip(goal_names, probs)}


def rounded(arr: np.ndarray, decimals: int = 3) -> np.ndarray:
    """Return rounded NumPy vector for display/tests."""

    return np.round(np.asarray(arr, dtype=float), decimals=decimals)
