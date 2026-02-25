from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .graph_utils import ArrayLike


@dataclass
class CSBMData:
    W: ArrayLike           # adjacency (float 0/1 for now)
    labels: ArrayLike      # +/-1 labels
    features: ArrayLike    # 1D contextual feature correlated with labels
    blocks: ArrayLike      # 0/1 block IDs


def generate_csbm(
    n: int = 200,
    p_in: float = 0.15,
    p_out: float = 0.03,
    mu: float = 1.0,
    sigma: float = 1.0,
    *,
    balanced: bool = True,
    seed: Optional[int] = None,
) -> CSBMData:
    """
    Generate a simplified 2-block Contextual SBM (CSBM):
    - Graph: undirected 2-block SBM with Bernoulli edges (p_in / p_out)
    - Labels: y_i in {-1,+1} tied to the two blocks
    - Context: 1D Gaussian features with mean mu * y_i and noise sigma

    This is a lightweight CSBM variant used for controlled experiments; it
    does not model higher-dimensional features or more complex link functions.

    Parameters
    ----------
    n : int
        Number of nodes.
    p_in, p_out : float
        Intra/inter-block edge probabilities.
    mu, sigma : float
        Context signal strength and noise std.
    balanced : bool
        If True, blocks are as balanced as possible.
    seed : int or None
        RNG seed.

    Returns
    -------
    CSBMData
    """
    if n <= 1:
        raise ValueError("n must be >= 2")
    if not (0 <= p_out <= 1 and 0 <= p_in <= 1):
        raise ValueError("p_in and p_out must lie in [0,1].")

    rng = np.random.default_rng(seed)

    if balanced:
        n1 = n // 2
        n2 = n - n1
        blocks = np.array([0] * n1 + [1] * n2, dtype=int)
        rng.shuffle(blocks)
    else:
        blocks = rng.integers(0, 2, size=n)

    labels = np.where(blocks == 0, -1.0, 1.0)

    # Sample undirected adjacency (no self-loops)
    W = np.zeros((n, n), dtype=float)
    iu = np.triu_indices(n, k=1)
    same_block = (blocks[iu[0]] == blocks[iu[1]])
    probs = np.where(same_block, p_in, p_out)
    edges = rng.random(size=probs.shape[0]) < probs
    W[iu] = edges.astype(float)
    W = W + W.T
    np.fill_diagonal(W, 0.0)

    # Contextual 1D feature
    eps = rng.normal(0.0, 1.0, size=n)
    features = mu * labels + sigma * eps

    return CSBMData(W=W, labels=labels, features=features, blocks=blocks)