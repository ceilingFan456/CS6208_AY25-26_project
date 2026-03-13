"""Bayesian posterior update utilities.

Same as the normal way
"""

from __future__ import annotations

import numpy as np


## Normalize our probability vector. 
def normalize(probs: np.ndarray) -> np.ndarray:
    total = float(np.sum(probs))
    if total <= 0.0:
        raise ValueError(f"Cannot normalize: probability mass is zero with prob vector {probs}")
    return probs / total


def posterior_update(prior: np.ndarray, likelihood: np.ndarray) -> np.ndarray:
    """ simple Bayesian vectorized update: posterior(g) ∝ likelihood(obs|g) * prior(g)."""

    unnormalized = prior * likelihood
    return normalize(unnormalized)
