from __future__ import annotations

import numpy as np

ArrayLike = np.ndarray


def validate_adjacency(W: ArrayLike, *, symmetrize: bool = False, zero_diag: bool = True) -> ArrayLike:
    """
    Validate (and optionally repair) an adjacency matrix W.

    Parameters
    ----------
    W : np.ndarray
        Square adjacency/weight matrix.
    symmetrize : bool
        If True, replace W by (W + W.T)/2.
    zero_diag : bool
        If True, force diagonal to zero.

    Returns
    -------
    np.ndarray
        Validated matrix (copy).
    """
    W = np.asarray(W, dtype=float).copy()
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be a square matrix.")
    if symmetrize:
        W = 0.5 * (W + W.T)
    if not np.allclose(W, W.T, atol=1e-12):
        raise ValueError("W must be symmetric (undirected graph).")
    if zero_diag:
        np.fill_diagonal(W, 0.0)
    if np.any(W < 0):
        raise ValueError("W must have nonnegative weights.")
    return W


def degree_vector(W: ArrayLike) -> ArrayLike:
    """Return degree vector d_i = sum_j w_ij."""
    W = validate_adjacency(W)
    return W.sum(axis=1)


def degree_matrix(W: ArrayLike) -> ArrayLike:
    """Return diagonal degree matrix D."""
    d = degree_vector(W)
    return np.diag(d)


def laplacian(W: ArrayLike) -> ArrayLike:
    """Return combinatorial Laplacian L = D - W."""
    W = validate_adjacency(W)
    d = W.sum(axis=1)
    return np.diag(d) - W


def normalized_laplacian(W: ArrayLike, eps: float = 1e-15) -> ArrayLike:
    """
    Return symmetric normalized Laplacian L_sym = I - D^{-1/2} W D^{-1/2}.

    Isolated nodes (degree ~ 0) get inverse sqrt degree 0.
    """
    W = validate_adjacency(W)
    d = W.sum(axis=1)
    inv_sqrt = np.zeros_like(d)
    mask = d > eps
    inv_sqrt[mask] = 1.0 / np.sqrt(d[mask])
    D_inv_sqrt = np.diag(inv_sqrt)
    n = W.shape[0]
    return np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt


def complete_graph_adjacency(n: int, weight: float = 1.0) -> ArrayLike:
    """
    Adjacency matrix of complete graph K_n with edge weight 'weight' and zero diagonal.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    W = np.full((n, n), float(weight))
    np.fill_diagonal(W, 0.0)
    return W


def edge_mass(W: ArrayLike) -> float:
    """
    Total undirected edge mass |E|_w = sum_{i<j} w_ij.
    """
    W = validate_adjacency(W)
    return float(np.triu(W, k=1).sum())


def number_of_pairs(n: int) -> int:
    """N = n(n-1)/2 unique unordered pairs."""
    if n < 0:
        raise ValueError("n must be nonnegative.")
    return n * (n - 1) // 2