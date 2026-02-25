from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .graph_utils import (
    ArrayLike,
    validate_adjacency,
    laplacian,
    complete_graph_adjacency,
    edge_mass,
    number_of_pairs,
)
from .signal_utils import as_signal, euclidean_norm_sq


def _upper_triangle_indices(n: int) -> Tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n, k=1)


def pairwise_differences(x: ArrayLike, *, absolute: bool = False, ordered: bool = False) -> ArrayLike:
    """
    Return pairwise differences x_i - x_j.

    Parameters
    ----------
    absolute : bool
        If True, return |x_i - x_j|.
    ordered : bool
        If True, return all ordered pairs i,j (including diagonal zeros).
        If False, return unique unordered pairs i<j.
    """
    x = as_signal(x)
    diff = x[:, None] - x[None, :]
    if absolute:
        diff = np.abs(diff)
    if ordered:
        return diff
    iu = _upper_triangle_indices(len(x))
    return diff[iu]


def gini_pairwise_numerator(x: ArrayLike, *, ordered: bool = False) -> float:
    """
    Unnormalized Gini-type pairwise numerator.

    If ordered=False:  sum_{i<j} |x_i - x_j|
    If ordered=True:   sum_{i,j}  |x_i - x_j|  (equals 2x unordered, diagonal contributes 0)
    """
    x = as_signal(x)
    if ordered:
        return float(np.abs(x[:, None] - x[None, :]).sum())
    return float(np.abs(pairwise_differences(x, absolute=False, ordered=False)).sum())


def weighted_total_variation(x: ArrayLike, W: ArrayLike) -> float:
    """
    Weighted graph total variation / weighted Gini-type numerator:

        TV_W(x) = sum_{i<j} w_ij |x_i - x_j|.
    """
    W = validate_adjacency(W)
    x = as_signal(x, n=W.shape[0])
    iu = _upper_triangle_indices(W.shape[0])
    return float((W[iu] * np.abs(x[iu[0]] - x[iu[1]])).sum())


def dirichlet_quadratic(x: ArrayLike, W: ArrayLike) -> float:
    """
    Dirichlet quadratic numerator:

        Q_W(x) = x^T L x = sum_{i<j} w_ij (x_i - x_j)^2
               = 1/2 sum_{i,j} w_ij (x_i - x_j)^2.
    """
    W = validate_adjacency(W)
    x = as_signal(x, n=W.shape[0])
    L = laplacian(W)
    return float(x @ L @ x)


def dirichlet_half_energy(x: ArrayLike, W: ArrayLike) -> float:
    """
    Alternative convention: E_half(x) = (1/2) * x^T L x.
    """
    return 0.5 * dirichlet_quadratic(x, W)


def dirichlet_pairwise_ordered_halfsum(x: ArrayLike, W: ArrayLike) -> float:
    """
    Explicitly compute (1/2) sum_{i,j} w_ij (x_i - x_j)^2 (sanity check for conventions).
    """
    W = validate_adjacency(W)
    x = as_signal(x, n=W.shape[0])
    diff = x[:, None] - x[None, :]
    return float(0.5 * np.sum(W * diff * diff))


def rayleigh_quotient(x: ArrayLike, W: ArrayLike, *, eps: float = 1e-15) -> float:
    """
    Unnormalized Laplacian Rayleigh quotient:
        R(x) = (x^T L x) / (x^T x)
    """
    x = as_signal(x)
    denom = euclidean_norm_sq(x)
    if denom <= eps:
        raise ValueError("x^T x is zero (or too close to zero).")
    return dirichlet_quadratic(x, W) / denom


def normalized_rayleigh_quotient(x: ArrayLike, W: ArrayLike, *, eps: float = 1e-15) -> float:
    """
    Normalized Laplacian-style quotient:
        R_norm(x) = (x^T L x) / (x^T D x)
    """
    from .signal_utils import weighted_norm_sq  # local import to avoid circularity concerns

    x = as_signal(x)
    denom = weighted_norm_sq(x, W)
    if denom <= eps:
        raise ValueError("x^T D x is zero (or too close to zero).")
    return dirichlet_quadratic(x, W) / denom


# ---------------------------------------------------------------------
# Complete graph identities and checks
# ---------------------------------------------------------------------

def complete_graph_dirichlet_identity_terms(x: ArrayLike) -> Dict[str, float]:
    """
    Return the quantities in the identity for K_n (unweighted complete graph):
        Q_{K_n}(x) = n * sum_i x_i^2 - (sum_i x_i)^2 = n * sum_i (x_i - mean)^2
    where Q_{K_n}(x) = sum_{i<j}(x_i - x_j)^2.

    Returns a dict with all terms and consistency checks.
    """
    x = as_signal(x)
    n = len(x)
    W = complete_graph_adjacency(n, weight=1.0)
    Q = dirichlet_quadratic(x, W)
    sum_x = float(x.sum())
    norm2 = euclidean_norm_sq(x)
    mean = float(x.mean())
    centered_ss = float(np.sum((x - mean) ** 2))

    rhs1 = float(n * norm2 - sum_x**2)
    rhs2 = float(n * centered_ss)

    return {
        "n": float(n),
        "Q_complete": Q,
        "n_times_norm2_minus_sumx2": rhs1,
        "n_times_centered_sumsq": rhs2,
        "abs_error_Q_rhs1": abs(Q - rhs1),
        "abs_error_Q_rhs2": abs(Q - rhs2),
        "sum_x": sum_x,
        "norm2": norm2,
        "mean_x": mean,
        "centered_sumsq": centered_ss,
    }


def centered_complete_graph_constant_check(x: ArrayLike, *, tol: float = 1e-9) -> Dict[str, float]:
    """
    For centered x on K_n, verifies Q_{K_n}(x) = n ||x||^2.

    If x is not centered, we still report the values and centeredness gap.
    """
    x = as_signal(x)
    n = len(x)
    W = complete_graph_adjacency(n)
    Q = dirichlet_quadratic(x, W)
    norm2 = euclidean_norm_sq(x)
    sum_x = float(x.sum())
    return {
        "n": float(n),
        "sum_x": sum_x,
        "Q_complete": Q,
        "n_norm2": float(n * norm2),
        "difference_Q_minus_n_norm2": float(Q - n * norm2),
        "is_centered_within_tol": float(abs(sum_x) <= tol),
    }


# ---------------------------------------------------------------------
# Cauchy-Schwarz bounds linking L1 (TV/Gini) and L2 (Dirichlet)
# ---------------------------------------------------------------------

def cauchy_schwarz_bound_complete(x: ArrayLike) -> Dict[str, float]:
    """
    Complete graph inequality (unordered-pair convention):
        G(x)^2 <= N * Q_{K_n}(x),    N = n(n-1)/2
    where
        G(x) = sum_{i<j} |x_i - x_j|,
        Q_{K_n}(x) = sum_{i<j} (x_i - x_j)^2.

    Returns values and slack.
    """
    x = as_signal(x)
    n = len(x)
    G = gini_pairwise_numerator(x, ordered=False)
    W = complete_graph_adjacency(n)
    Q = dirichlet_quadratic(x, W)
    N = float(number_of_pairs(n))
    bound_sq = N * Q
    return {
        "n": float(n),
        "num_pairs": N,
        "gini_num_unordered": G,
        "dirichlet_Q_complete": Q,
        "lhs_gini_sq": float(G**2),
        "rhs_bound_sq": float(bound_sq),
        "holds": float(G**2 <= bound_sq + 1e-10),
        "slack_rhs_minus_lhs": float(bound_sq - G**2),
        "bound_on_gini": float(np.sqrt(max(bound_sq, 0.0))),
    }


def cauchy_schwarz_bound_complete_centered_to_norm(x: ArrayLike) -> Dict[str, float]:
    """
    If x is centered on K_n, then G(x) <= n * sqrt((n-1)/2) * ||x||.

    This function computes the RHS using the centered identity Q = n||x||^2,
    but always reports the actual centeredness.
    """
    x = as_signal(x)
    n = len(x)
    G = gini_pairwise_numerator(x, ordered=False)
    norm = float(np.linalg.norm(x))
    rhs = float(n * np.sqrt((n - 1) / 2.0) * norm)
    return {
        "n": float(n),
        "sum_x": float(x.sum()),
        "gini_num_unordered": G,
        "norm_x": norm,
        "rhs_centered_bound": rhs,
        "holds_if_centered": float(G <= rhs + 1e-10),
    }


def cauchy_schwarz_bound_weighted(
    x: ArrayLike, W: ArrayLike, *, half_energy_convention: bool = False
) -> Dict[str, float]:
    """
    Weighted graph inequality:
        TV_W(x)^2 <= |E|_w * Q_W(x)

    If half_energy_convention=True, also report the equivalent form with
    E_half = (1/2)Q_W:
        TV_W(x)^2 <= 2|E|_w * E_half(x)
    """
    W = validate_adjacency(W)
    x = as_signal(x, n=W.shape[0])

    TV = weighted_total_variation(x, W)
    Q = dirichlet_quadratic(x, W)
    Ew = edge_mass(W)
    rhs = Ew * Q

    out = {
        "tv_weighted": TV,
        "dirichlet_Q": Q,
        "edge_mass": Ew,
        "lhs_tv_sq": float(TV**2),
        "rhs_bound_sq": float(rhs),
        "holds": float(TV**2 <= rhs + 1e-10),
        "slack_rhs_minus_lhs": float(rhs - TV**2),
        "bound_on_tv": float(np.sqrt(max(rhs, 0.0))),
    }
    if half_energy_convention:
        out["dirichlet_half_energy"] = 0.5 * Q
        out["rhs_bound_sq_half_energy_form"] = float(2.0 * Ew * (0.5 * Q))
    return out


# ---------------------------------------------------------------------
# Gradients / subgradients
# ---------------------------------------------------------------------

def dirichlet_gradient(x: ArrayLike, W: ArrayLike, *, half_scaled: bool = False) -> ArrayLike:
    """
    Gradient of the quadratic Dirichlet objective.

    If half_scaled=False:
        objective = Q_W(x) = x^T L x
        grad = 2 L x

    If half_scaled=True:
        objective = (1/2) Q_W(x)
        grad = L x
    """
    W = validate_adjacency(W)
    x = as_signal(x, n=W.shape[0])
    L = laplacian(W)
    g = L @ x
    return g if half_scaled else 2.0 * g


def tv_subgradient(x: ArrayLike, W: ArrayLike, *, tie_value: float = 0.0) -> ArrayLike:
    """
    A valid subgradient selection for TV_W(x) = sum_{i<j} w_ij |x_i-x_j|.

    For ties x_i == x_j, sgn(0) can be any value in [-1,1]. This function uses tie_value
    (default 0.0), clipped to [-1,1].

    Returns a vector g such that one subgradient is:
        g_i = sum_j w_ij * sgn(x_i - x_j)
    """
    W = validate_adjacency(W)
    x = as_signal(x, n=W.shape[0])
    t = float(np.clip(tie_value, -1.0, 1.0))
    diff = x[:, None] - x[None, :]
    s = np.sign(diff)
    s[np.isclose(diff, 0.0)] = t
    g = np.sum(W * s, axis=1)
    return g