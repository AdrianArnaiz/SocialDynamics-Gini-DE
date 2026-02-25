from __future__ import annotations

from typing import Optional

import numpy as np

from .graph_utils import ArrayLike, degree_vector


def as_signal(x: ArrayLike, n: Optional[int] = None) -> ArrayLike:
    """
    Validate/reshape x as a length-n 1D float vector.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    if n is not None and x.shape[0] != n:
        raise ValueError(f"x must have length {n}, got {x.shape[0]}")
    return x


def center_signal(x: ArrayLike) -> ArrayLike:
    """
    Return x - mean(x).
    """
    x = as_signal(x)
    return x - x.mean()


def euclidean_norm_sq(x: ArrayLike) -> float:
    """
    Return ||x||^2 = x^T x.
    """
    x = as_signal(x)
    return float(x @ x)


def weighted_norm_sq(x: ArrayLike, W: ArrayLike) -> float:
    """
    Return x^T D x, where D is degree matrix of W.
    """
    x = as_signal(x)
    d = degree_vector(W)
    if len(x) != len(d):
        raise ValueError("x and W dimension mismatch.")
    return float(np.dot(d * x, x))