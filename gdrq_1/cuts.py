from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from .graph_utils import ArrayLike, validate_adjacency, degree_vector
from .signal_utils import as_signal


def cut_value(S: Sequence[int], W: ArrayLike) -> float:
    """
    Weighted cut value Cut(S, Sbar) = sum_{i in S, j not in S} w_ij.
    """
    W = validate_adjacency(W)
    n = W.shape[0]
    S_mask = np.zeros(n, dtype=bool)
    S_mask[np.asarray(S, dtype=int)] = True
    Sc_mask = ~S_mask
    return float(W[np.ix_(S_mask, Sc_mask)].sum())


def volume(S: Sequence[int], W: ArrayLike) -> float:
    """
    Volume Vol(S) = sum_{i in S} d_i.
    """
    W = validate_adjacency(W)
    d = degree_vector(W)
    idx = np.asarray(S, dtype=int)
    return float(d[idx].sum())


def conductance(S: Sequence[int], W: ArrayLike, *, eps: float = 1e-15) -> float:
    """
    Conductance-style Cheeger objective:
        phi(S) = Cut(S,Sbar) / min(Vol(S), Vol(Sbar))
    """
    W = validate_adjacency(W)
    n = W.shape[0]
    S = np.unique(np.asarray(S, dtype=int))
    if len(S) == 0 or len(S) == n:
        return float("inf")

    all_idx = np.arange(n)
    S_mask = np.zeros(n, dtype=bool)
    S_mask[S] = True
    Sc = all_idx[~S_mask]

    cut = cut_value(S, W)
    volS = volume(S, W)
    volSc = volume(Sc, W)
    denom = min(volS, volSc)
    if denom <= eps:
        return float("inf")
    return float(cut / denom)


def sweep_cut_from_scores(scores: ArrayLike, W: ArrayLike) -> Dict[str, object]:
    """
    Basic sweep-cut over sorted scores. Returns the best set by conductance.
    """
    W = validate_adjacency(W)
    scores = as_signal(scores, n=W.shape[0])

    order = np.argsort(scores)
    best_phi = float("inf")
    best_k = None
    best_set = None

    for k in range(1, len(scores)):
        S = order[:k]
        phi = conductance(S, W)
        if phi < best_phi:
            best_phi = phi
            best_k = k
            best_set = S.copy()

    return {
        "best_conductance": float(best_phi),
        "best_k": int(best_k) if best_k is not None else None,
        "best_set": best_set.tolist() if best_set is not None else None,
        "sorted_order": order.tolist(),
    }