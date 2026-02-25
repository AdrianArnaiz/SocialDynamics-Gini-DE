"""
rq_gi_dynamics.py
=================

A compact, reproducible sandbox to explore the relationship between:

- Dirichlet energy / Laplacian quadratic form:      DE(x;G) = x^T L x
- Rayleigh quotient (one common version):          RQ_L2(x;G) = (x^T L x) / (x^T x)
- Dispersion proxies (variance, MAD) and Gini-like L1 dispersion measures

and how they evolve under simple opinion dynamics:
- DeGroot averaging (row-stochastic linear dynamics) -> convergence
- Bounded-confidence / intolerance (only listen to sufficiently similar neighbors) -> persistent polarization

This script is intentionally organized into:
1) Calculation (metrics, Laplacian, utilities)
2) Simulation (dynamics)
3) Scenario logic (graph + opinion setups)
4) Experiment launcher (run_all_experiments)

Run:
    python rq_gi_dynamics.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


# =============================================================================
# 1) CALCULATION
# =============================================================================

def compute_laplacian(G: nx.Graph) -> np.ndarray:
    """Return the (dense) combinatorial Laplacian L = D - A."""
    return nx.laplacian_matrix(G).toarray().astype(float)


def dirichlet_energy_from_laplacian(L: np.ndarray, x: np.ndarray) -> float:
    """DE(x;G) = x^T L x (for undirected graphs: sum_{(i,j) in E} (x_i - x_j)^2)."""
    x = np.asarray(x, dtype=float)
    return float(x.T @ L @ x)


def gini_nonnegative(x: np.ndarray, eps: float = 1e-12) -> float:
    """
    Standard Gini coefficient for nonnegative data:
        G = mean_{i,j} |x_i - x_j| / (2 * mean(x))

    Raises ValueError if mean(x) <= 0 or any x_i < 0.
    """
    x = np.asarray(x, dtype=float)
    if np.any(x < 0):
        raise ValueError("gini_nonnegative requires x >= 0.")
    mu = float(np.mean(x))
    if mu <= eps:
        raise ValueError("gini_nonnegative requires mean(x) > 0.")
    abs_diff = np.abs(np.subtract.outer(x, x))
    return float(abs_diff.mean() / (2.0 * mu))


def gini_signed(x: np.ndarray, eps: float = 1e-12) -> float:
    """
    A robust 'Gini-like' relative mean absolute difference that works with signed opinions:
        G_signed = mean_{i,j} |x_i - x_j| / (2 * mean(|x|))

    This is not the classical income-inequality Gini (which assumes nonnegative),
    but keeps the same L1-pairwise numerator and a positive scale denominator.
    """
    x = np.asarray(x, dtype=float)
    denom = float(np.mean(np.abs(x)))
    if denom <= eps:
        return float("nan")
    abs_diff = np.abs(np.subtract.outer(x, x))
    return float(abs_diff.mean() / (2.0 * denom))


def mean_abs_deviation(x: np.ndarray) -> float:
    """MAD = mean_i |x_i - mean(x)|."""
    x = np.asarray(x, dtype=float)
    return float(np.mean(np.abs(x - np.mean(x))))


def safe_div(a: float, b: float, eps: float = 1e-12) -> float:
    return float(a / b) if abs(b) > eps else float("nan")


def get_metrics_full(
    G: nx.Graph,
    x: np.ndarray,
    L: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    Unified metrics in one shot (Laplacian computed once).

    Returns a dict with:
    - de                  : x^T L x
    - l2_norm2            : x^T x
    - rq_l2               : (x^T L x) / (x^T x)
    - mean, var, mad
    - rq_var              : (x^T L x) / var(x)     (when var > 0)
    - gini_nonneg (or nan): classical Gini if x >= 0 and mean(x) > 0
    - gini_signed         : robust Gini-like for signed signals
    """
    x = np.asarray(x, dtype=float)
    if L is None:
        L = compute_laplacian(G)

    de = dirichlet_energy_from_laplacian(L, x)
    l2 = float(x.T @ x)
    mu = float(np.mean(x))
    var = float(np.var(x))
    mad = mean_abs_deviation(x)

    # Gini variants
    try:
        g_nonneg = gini_nonnegative(x, eps=eps)
    except ValueError:
        g_nonneg = float("nan")
    g_signed = gini_signed(x, eps=eps)

    return {
        "n": float(G.number_of_nodes()),
        "m": float(G.number_of_edges()),
        "mean": mu,
        "var": var,
        "mad": mad,
        "gini_nonneg": g_nonneg,
        "gini_signed": g_signed,
        "de": de,
        "l2_norm2": l2,
        "rq_l2": safe_div(de, l2, eps=eps),
        "rq_var": safe_div(de, var, eps=eps),
    }


# =============================================================================
# 2) SIMULATION
# =============================================================================

def build_row_stochastic_matrix(G: nx.Graph, self_loops: bool = True) -> np.ndarray:
    """
    Build a DeGroot matrix W from adjacency:
        W = D^{-1} (A + I)  if self_loops else D^{-1} A

    Handles isolated nodes by putting all mass on self.
    """
    A = nx.to_numpy_array(G, dtype=float)
    if self_loops:
        A = A + np.eye(G.number_of_nodes(), dtype=float)

    row_sums = A.sum(axis=1)
    W = np.zeros_like(A)

    # Avoid division by zero for isolated nodes
    for i, s in enumerate(row_sums):
        if s > 0:
            W[i, :] = A[i, :] / s
        else:
            W[i, i] = 1.0
    return W


@dataclass
class SimulationHistory:
    x_hist: np.ndarray            # shape: (T+1, n)
    metrics_hist: List[Dict[str, float]]


def simulate_degroot(
    G: nx.Graph,
    x0: np.ndarray,
    steps: int = 20,
    self_loops: bool = True,
) -> SimulationHistory:
    """
    Linear DeGroot averaging:
        x_{t+1} = W x_t, with W row-stochastic.
    """
    x = np.asarray(x0, dtype=float).copy()
    W = build_row_stochastic_matrix(G, self_loops=self_loops)
    L = compute_laplacian(G)

    x_hist = [x.copy()]
    metrics_hist = [get_metrics_full(G, x, L=L)]

    for _ in range(steps):
        x = W @ x
        x_hist.append(x.copy())
        metrics_hist.append(get_metrics_full(G, x, L=L))

    return SimulationHistory(x_hist=np.vstack(x_hist), metrics_hist=metrics_hist)


def simulate_bounded_confidence(
    G: nx.Graph,
    x0: np.ndarray,
    threshold: float = 0.3,
    steps: int = 50,
    include_self: bool = True,
) -> SimulationHistory:
    """
    A simple bounded-confidence / intolerance rule:
      node i averages only neighbors j with |x_i - x_j| < threshold.
    Optionally includes itself always.

    Note: This is not exactly HK / Deffuant, but captures the 'only listen to close opinions' effect.
    """
    x = np.asarray(x0, dtype=float).copy()
    n = len(x)
    adj = [list(G.neighbors(i)) for i in range(n)]
    L = compute_laplacian(G)

    x_hist = [x.copy()]
    metrics_hist = [get_metrics_full(G, x, L=L)]

    for _ in range(steps):
        new_x = np.empty_like(x)
        for i in range(n):
            neigh = adj[i]
            close = [j for j in neigh if abs(x[i] - x[j]) < threshold]
            voices = close + ([i] if include_self else [])
            if voices:
                new_x[i] = float(np.mean(x[voices]))
            else:
                new_x[i] = x[i]
        x = new_x
        x_hist.append(x.copy())
        metrics_hist.append(get_metrics_full(G, x, L=L))

    return SimulationHistory(x_hist=np.vstack(x_hist), metrics_hist=metrics_hist)


# =============================================================================
# 3) VISUALIZATION
# =============================================================================

def _group_means(x_hist: np.ndarray, split: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mean_A[t], mean_B[t]) where A = [0:split), B = [split:n)."""
    mean_A = x_hist[:, :split].mean(axis=1)
    mean_B = x_hist[:, split:].mean(axis=1)
    return mean_A, mean_B


def plot_opinion_dynamics_comparison(
    degroot: SimulationHistory,
    bounded: SimulationHistory,
    split: int,
    title: str = "Opinion dynamics: DeGroot vs. Bounded-confidence",
) -> None:
    """
    One comparative graph:
    - solid lines: DeGroot group means
    - dashed lines: bounded-confidence group means
    """
    t1 = np.arange(degroot.x_hist.shape[0])
    t2 = np.arange(bounded.x_hist.shape[0])

    dA, dB = _group_means(degroot.x_hist, split)
    bA, bB = _group_means(bounded.x_hist, split)

    plt.figure(figsize=(10, 5))
    plt.plot(t1, dA, label="DeGroot: mean(A)")
    plt.plot(t1, dB, label="DeGroot: mean(B)")
    plt.plot(t2, bA, linestyle="--", label="Bounded: mean(A)")
    plt.plot(t2, bB, linestyle="--", label="Bounded: mean(B)")
    plt.title(title)
    plt.xlabel("Time (iterations)")
    plt.ylabel("Opinion")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_metric_comparison(
    degroot: SimulationHistory,
    bounded: SimulationHistory,
    key: str = "rq_l2",
    title: Optional[str] = None,
) -> None:
    """Compare a single metric time-series for two dynamics."""
    y1 = np.array([m[key] for m in degroot.metrics_hist], dtype=float)
    y2 = np.array([m[key] for m in bounded.metrics_hist], dtype=float)
    t1 = np.arange(len(y1))
    t2 = np.arange(len(y2))

    plt.figure(figsize=(10, 4))
    plt.plot(t1, y1, label=f"DeGroot: {key}")
    plt.plot(t2, y2, linestyle="--", label=f"Bounded: {key}")
    plt.title(title or f"Metric comparison: {key}")
    plt.xlabel("Time (iterations)")
    plt.ylabel(key)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =============================================================================
# 4) SCENARIO LOGIC (graphs + initial opinions)
# =============================================================================

def scenario_segregation_vs_integration(
    n_per_group: int = 50,
    p_in: float = 0.5,
    p_out_segregated: float = 0.01,
    p_out_integrated: float = 0.2,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Two planted-partition graphs with identical opinion distribution but different mixing.
    Shows: Gini (on positive-coded opinions) stays constant, while DE/RQ changes with topology.
    """
    rng = np.random.default_rng(seed)

    n = 2 * n_per_group

    # For Gini-as-inequality, keep positive coding.
    x_gini = np.array([1.0] * n_per_group + [2.0] * n_per_group)
    # For tension across camps, a centered signal is convenient.
    x_centered = np.array([1.0] * n_per_group + [-1.0] * n_per_group)

    G_segr = nx.planted_partition_graph(2, n_per_group, p_in=p_in, p_out=p_out_segregated, seed=int(rng.integers(1e9)))
    G_integ = nx.planted_partition_graph(2, n_per_group, p_in=p_in, p_out=p_out_integrated, seed=int(rng.integers(1e9)))

    m1 = get_metrics_full(G_segr, x_centered)
    m2 = get_metrics_full(G_integ, x_centered)

    gi = gini_nonnegative(x_gini)

    df = pd.DataFrame(
        [
            {
                "scenario": "segregated",
                "gini_nonneg(opinions 1/2)": gi,
                "de(centered +/-1)": m1["de"],
                "rq_l2(centered)": m1["rq_l2"],
            },
            {
                "scenario": "integrated",
                "gini_nonneg(opinions 1/2)": gi,
                "de(centered +/-1)": m2["de"],
                "rq_l2(centered)": m2["rq_l2"],
            },
        ]
    )
    df["rq_increase_%"] = 100.0 * (df["rq_l2(centered)"] / df.loc[0, "rq_l2(centered)"] - 1.0)
    return df


def scenario_four_quadrants(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """
    Four toy 'social states' (illustrative, not canonical):
      1) Consensus: low dispersion, low DE
      2) Echo chamber: high dispersion, low DE (segregated topology)
      3) Conflict (integrated): high dispersion, high DE
      4) Friction/noise: low-ish dispersion but high DE on dense graph
    """
    rng = np.random.default_rng(seed)
    n2 = n // 2

    # 1) Consensus
    x_cons = rng.normal(0.5, 0.05, n)
    G_cons = nx.erdos_renyi_graph(n, 0.1, seed=int(rng.integers(1e9)))

    # 2) Echo chamber (two camps, almost disconnected)
    x_echo = np.array([1.0] * n2 + [-1.0] * (n - n2))
    G_echo = nx.planted_partition_graph(2, n2, p_in=0.2, p_out=0.001, seed=int(rng.integers(1e9)))

    # 3) Integrated conflict (two camps, well mixed)
    x_conf = np.array([1.0] * n2 + [-1.0] * (n - n2))
    G_conf = nx.planted_partition_graph(2, n2, p_in=0.1, p_out=0.1, seed=int(rng.integers(1e9)))

    # 4) Friction/noise on dense graph
    x_noise = rng.normal(0.0, 0.1, n)
    G_noise = nx.complete_graph(n)

    scenarios = [
        ("Consensus", G_cons, x_cons),
        ("Echo chamber", G_echo, x_echo),
        ("Integrated conflict", G_conf, x_conf),
        ("Friction/noise (dense)", G_noise, x_noise),
    ]

    rows = []
    for name, G, x in scenarios:
        m = get_metrics_full(G, x)
        rows.append(
            {
                "state": name,
                "de": m["de"],
                "var": m["var"],
                "rq_l2": m["rq_l2"],
                "rq_var(de/var)": m["rq_var"],
                "gini_signed": m["gini_signed"],
                "edges": int(m["m"]),
            }
        )
    return pd.DataFrame(rows)


def scenario_radicalization_path(n: int = 100, radical_strength: float = 2.0, seed: int = 42) -> pd.DataFrame:
    """
    A 3-step path:
      1) Consensus-ish
      2) Inject conflict by shifting half the nodes
      3) 'Fracture' by removing many cross edges between the two halves
    """
    rng = np.random.default_rng(seed)

    # Step 1: initial consensus-ish
    x1 = rng.normal(0.0, 0.1, n)
    G1 = nx.erdos_renyi_graph(n, 0.1, seed=int(rng.integers(1e9)))

    # Step 2: radicalize half toward +radical_strength
    x2 = x1.copy()
    x2[: n // 2] += radical_strength
    G2 = G1

    # Step 3: fracture by removing most edges between halves
    G3 = G1.copy()
    edges_cross = [
        (u, v)
        for (u, v) in list(G3.edges())
        if (u < n // 2 and v >= n // 2) or (u >= n // 2 and v < n // 2)
    ]
    if edges_cross:
        k = int(0.9 * len(edges_cross))
        to_remove = rng.choice(len(edges_cross), size=k, replace=False)
        G3.remove_edges_from([edges_cross[i] for i in to_remove])

    rows = []
    for label, G, x in [
        ("1. Consensus", G1, x1),
        ("2. Conflict (radicalization)", G2, x2),
        ("3. Echo chamber (fracture)", G3, x2),
    ]:
        m = get_metrics_full(G, x)
        rows.append(
            {
                "state": label,
                "de": m["de"],
                "var": m["var"],
                "rq_l2": m["rq_l2"],
                "rq_var(de/var)": m["rq_var"],
                "gini_signed": m["gini_signed"],
                "edges": int(m["m"]),
            }
        )
    return pd.DataFrame(rows)


def _break_edges_fraction(G: nx.Graph, frac_remove: float, seed: int) -> nx.Graph:
    rng = np.random.default_rng(seed)
    H = G.copy()
    edges = list(H.edges())
    k = int(frac_remove * len(edges))
    if k > 0:
        idx = rng.choice(len(edges), size=k, replace=False)
        H.remove_edges_from([edges[i] for i in idx])
    return H


def scenario_mediation(
    n_per_side: int = 50,
    n_mediators: int = 20,
    mediator_opinion: float = 1.0,
    frac_remove: float = 0.9,
    degree_to_each_side: int = 5,
    seed: int = 42,
) -> Tuple[pd.DataFrame, nx.Graph, np.ndarray]:
    """
    Mediation scenario on opinions in [0,2]:
    - Two sides: A at 0, B at 2
    - Start from complete bipartite (max cross conflict, no within-camp edges)
    - Fracture by removing a fraction of edges
    - Add mediator nodes with opinion mediator_opinion, connected to degree_to_each_side nodes in each camp

    Returns:
    - DataFrame with metrics for (Conflict, Fracture, Mediation)
    - the mediated graph, and its opinion vector (useful for dynamics)
    """
    rng = np.random.default_rng(seed)

    # Conflict graph: complete bipartite between camps.
    x_conf = np.array([0.0] * n_per_side + [2.0] * n_per_side)
    G_conf = nx.complete_multipartite_graph(n_per_side, n_per_side)

    # Fracture: remove many cross edges
    G_broken = _break_edges_fraction(G_conf, frac_remove=frac_remove, seed=int(rng.integers(1e9)))

    # Mediation: add mediator nodes that connect to both sides
    G_med = G_broken.copy()
    x_med = x_conf.tolist()

    mediator_start = 2 * n_per_side
    for i in range(n_mediators):
        node = mediator_start + i
        G_med.add_node(node)
        x_med.append(float(mediator_opinion))

        targets_A = rng.choice(n_per_side, size=degree_to_each_side, replace=False).tolist()
        targets_B = rng.choice(range(n_per_side, 2 * n_per_side), size=degree_to_each_side, replace=False).tolist()
        for t in targets_A + targets_B:
            G_med.add_edge(node, t)

    x_med = np.array(x_med, dtype=float)

    # Metrics
    rows = []
    for label, G, x in [
        ("Conflict (complete bipartite)", G_conf, x_conf),
        ("Fracture (removed edges)", G_broken, x_conf),
        (f"Mediation (mediators at {mediator_opinion})", G_med, x_med),
    ]:
        m = get_metrics_full(G, x)
        rows.append(
            {
                "state": label,
                "de": m["de"],
                "var": m["var"],
                "rq_l2": m["rq_l2"],
                "rq_var(de/var)": m["rq_var"],
                "gini_nonneg": m["gini_nonneg"],  # valid here (x >= 0)
                "edges": int(m["m"]),
                "nodes": int(m["n"]),
            }
        )

    return pd.DataFrame(rows), G_med, x_med


# =============================================================================
# 5) EXPERIMENT LAUNCHER
# =============================================================================

def run_all_experiments(
    seed: int = 42,
    show_plots: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Entry point that runs all toy experiments and returns their DataFrames
    (plus plots if show_plots=True).
    """
    results: Dict[str, pd.DataFrame] = {}

    # --- Experiment 1: RQ changes with topology while Gini on opinions stays fixed
    df_topo = scenario_segregation_vs_integration(seed=seed)
    results["topology_effect"] = df_topo
    print("\n=== Experiment 1: Topology effect (Gini fixed, RQ changes) ===")
    print(df_topo.to_string(index=False))

    # --- Experiment 2: Dynamics comparison on an integrated graph
    # Build a mixed graph similar to the 'integrated' case:
    n_per = 50
    G2 = nx.planted_partition_graph(2, n_per, p_in=0.5, p_out=0.2, seed=seed)
    x0 = np.array([1.0] * n_per + [-1.0] * n_per)

    deg = simulate_degroot(G2, x0, steps=25)
    bc = simulate_bounded_confidence(G2, x0, threshold=0.3, steps=60)

    if show_plots:
        plot_opinion_dynamics_comparison(deg, bc, split=n_per)
        plot_metric_comparison(deg, bc, key="rq_l2", title="RQ_L2 over time: convergence vs persistent polarization")

    # Provide a compact summary table
    def _final_row(hist: SimulationHistory, name: str) -> Dict[str, float]:
        m = hist.metrics_hist[-1]
        return {
            "dynamics": name,
            "final_mean": m["mean"],
            "final_var": m["var"],
            "final_de": m["de"],
            "final_rq_l2": m["rq_l2"],
            "final_gini_signed": m["gini_signed"],
        }

    df_dyn = pd.DataFrame([_final_row(deg, "DeGroot"), _final_row(bc, "Bounded-confidence")])
    results["dynamics_summary"] = df_dyn
    print("\n=== Experiment 2: Dynamics summary (final state) ===")
    print(df_dyn.to_string(index=False))

    # --- Experiment 3: Four-quadrant toy map
    df_quad = scenario_four_quadrants(seed=seed)
    results["four_quadrants"] = df_quad
    print("\n=== Experiment 3: Four toy social states ===")
    print(df_quad.to_string(index=False))

    # --- Experiment 4: Radicalization -> fracture
    df_rad = scenario_radicalization_path(seed=seed)
    results["radicalization_path"] = df_rad
    print("\n=== Experiment 4: Radicalization path ===")
    print(df_rad.to_string(index=False))

    # --- Experiment 5: Mediation (neutral vs biased mediators)
    df_med_neutral, G_med_neutral, x_med_neutral = scenario_mediation(mediator_opinion=1.0, seed=seed)
    df_med_biased, G_med_biased, x_med_biased = scenario_mediation(mediator_opinion=1.8, seed=seed + 1)

    df_med_compare = pd.concat(
        [
            df_med_neutral.assign(case="neutral_mediators"),
            df_med_biased.assign(case="biased_mediators"),
        ],
        ignore_index=True,
    )
    results["mediation_compare"] = df_med_compare
    print("\n=== Experiment 5: Mediation (neutral vs biased) ===")
    print(df_med_compare.to_string(index=False))

    # Optional: show how biased mediators change dynamics (DeGroot) on mediated graphs
    if show_plots:
        # Re-map to signed axis to compare (shift [0,2] -> [-1,1])
        xN = x_med_neutral - 1.0
        xB = x_med_biased - 1.0

        degN = simulate_degroot(G_med_neutral, xN, steps=35)
        degB = simulate_degroot(G_med_biased, xB, steps=35)

        # Plot the mean of A and B camps only (ignore mediator nodes for group split)
        split = 50
        plt.figure(figsize=(10, 4))
        t = np.arange(degN.x_hist.shape[0])
        aN, bN = _group_means(degN.x_hist[:, : 100], split=split)  # only original 100 nodes
        aB, bB = _group_means(degB.x_hist[:, : 100], split=split)
        plt.plot(t, aN, label="Neutral mediators: mean(A)")
        plt.plot(t, bN, label="Neutral mediators: mean(B)")
        plt.plot(t, aB, linestyle="--", label="Biased mediators: mean(A)")
        plt.plot(t, bB, linestyle="--", label="Biased mediators: mean(B)")
        plt.axhline(0.0, color="gray", linestyle=":", label="Center (0)")
        plt.title("DeGroot with mediators: neutral vs biased (camp means)")
        plt.xlabel("Time (iterations)")
        plt.ylabel("Opinion (shifted to [-1,1])")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return results


if __name__ == "__main__":
    run_all_experiments(seed=42, show_plots=True)