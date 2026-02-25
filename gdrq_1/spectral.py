from __future__ import annotations

from typing import Tuple

import numpy as np

from .graph_utils import ArrayLike, validate_adjacency, laplacian, normalized_laplacian


def fiedler_vector(W: ArrayLike, *, normalized: bool = False) -> Tuple[float, ArrayLike]:
    """
    Return (lambda2, v2) for the graph Laplacian.

    Parameters
    ----------
    normalized : bool
        If True, use symmetric normalized Laplacian.
        If False, use combinatorial Laplacian.

    Notes
    -----
    For disconnected graphs, lambda2 may be ~0 and the vector is not unique.
    """
    W = validate_adjacency(W)
    L = normalized_laplacian(W) if normalized else laplacian(W)
    evals, evecs = np.linalg.eigh(L)
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]
    if len(evals) < 2:
        return float(evals[0]), evecs[:, 0].copy()
    return float(evals[1]), evecs[:, 1].copy()