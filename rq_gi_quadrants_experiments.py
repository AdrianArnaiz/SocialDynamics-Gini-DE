"""
rq_gi_quadrants_experiments.py
==============================

Experiments to operationalize the "4 quadrants" narrative:

Quadrants in (Diversity, Tension) space, where:
- Diversity: variance of opinions (absolute dispersion)
- Tension:  Rayleigh quotient RQ_L2 = (x^T L x) / (x^T x)

We include:
1) A quadrant map (4 archetypes) + a topology sweep (same diversity, different tension)
2) Transition trajectories (radicalization -> fracture -> mediation), with neutral vs biased mediators
3) Dynamics trajectories (DeGroot vs bounded-confidence) in the quadrant plane + camp means
4) Phase diagram over (cross-group mixing p_out, intolerance threshold), showing final polarization and final tension

Run:
    python rq_gi_quadrants_experiments.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


# =============================================================================
# 1) METRICS
# =============================================================================

def compute_laplacian(G: nx.Graph) -> np.ndarray:
    return nx.laplacian_matrix(G).toarray().astype(float)


def safe_div(a: float, b: float, eps: float = 1e-12) -> float:
    return float(a / b) if abs(b) > eps else float("nan")


def gini_nonnegative(x: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=float)
    if np.any(x < 0):
        raise ValueError("Needs x >= 0.")
    mu = float(np.mean(x))
    if mu <= eps:
        raise ValueError("Needs mean(x) > 0.")
    abs_diff = np.abs(np.subtract.outer(x, x))
    return float(abs_diff.mean() / (2.0 * mu))


def gini_signed(x: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=float)
    denom = float(np.mean(np.abs(x)))
    if denom <= eps:
        return float("nan")
    abs_diff = np.abs(np.subtract.outer(x, x))
    return float(abs_diff.mean() / (2.0 * denom))


def get_metrics_full(G: nx.Graph, x: np.ndarray, L: Optional[np.ndarray] = None) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    if L is None:
        L = compute_laplacian(G)

    de = float(x.T @ L @ x)
    l2 = float(x.T @ x)
    var = float(np.var(x))
    try:
        g_nonneg = gini_nonnegative(x)
    except Exception:
        g_nonneg = float("nan")
    g_signed = gini_signed(x)

    m = float(G.number_of_edges())
    return {
        "n": float(G.number_of_nodes()),
        "m": m,
        "mean": float(np.mean(x)),
        "var": var,
        "gini_nonneg": g_nonneg,
        "gini_signed": g_signed,
        "de": de,
        "rq_l2": safe_div(de, l2),
        "rq_var": safe_div(de, var),
        "de_per_edge": safe_div(de, m),
    }


# =============================================================================
# 2) DYNAMICS
# =============================================================================

@dataclass
class SimulationHistory:
    x_hist: np.ndarray
    metrics_hist: List[Dict[str, float]]


def build_row_stochastic_matrix(G: nx.Graph, self_loops: bool = True) -> np.ndarray:
    A = nx.to_numpy_array(G, dtype=float)
    if self_loops:
        A = A + np.eye(G.number_of_nodes(), dtype=float)
    row_sums = A.sum(axis=1)
    W = np.zeros_like(A)
    for i, s in enumerate(row_sums):
        if s > 0:
            W[i, :] = A[i, :] / s
        else:
            W[i, i] = 1.0
    return W


def simulate_degroot(G: nx.Graph, x0: np.ndarray, steps: int = 30) -> SimulationHistory:
    x = np.asarray(x0, dtype=float).copy()
    W = build_row_stochastic_matrix(G, self_loops=True)
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
    threshold: float = 0.35,
    steps: int = 80,
) -> SimulationHistory:
    """
    Bounded-confidence / intolerance:
      i averages only neighbors with |x_i - x_j| <= threshold, plus itself.

    Note: For pure +/-1 camps, cross-camp distance is ~2, so threshold must exceed ~2
    to allow cross-listening without mediators or noise.
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
            close = [j for j in adj[i] if abs(x[i] - x[j]) <= threshold]
            voices = close + [i]  # include self
            new_x[i] = float(np.mean(x[voices])) if voices else x[i]
        x = new_x
        x_hist.append(x.copy())
        metrics_hist.append(get_metrics_full(G, x, L=L))

    return SimulationHistory(x_hist=np.vstack(x_hist), metrics_hist=metrics_hist)


def group_means(x_hist: np.ndarray, split: int) -> Tuple[np.ndarray, np.ndarray]:
    return x_hist[:, :split].mean(axis=1), x_hist[:, split:].mean(axis=1)


# =============================================================================
# 3) SCENARIOS
# =============================================================================

def make_two_camp_signal(n_per_side: int = 50, low: float = -1.0, high: float = 1.0, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.array([high]*n_per_side + [low]*n_per_side, dtype=float)
    if noise > 0:
        x = x + rng.normal(0.0, noise, size=x.shape[0])
    return x


def quadrant_archetypes(seed: int = 42, n: int = 100) -> pd.DataFrame:
    """
    Four archetypes, designed to land in the intended quadrants when using:
      x-axis: var(x)   (GI/Var proxy for diversity)
      y-axis: rq_l2    (Dirichlet tension normalized by L2 energy)
    """
    rng = np.random.default_rng(seed)
    n2 = n // 2

    # Low diversity, low tension: mild consensus on sparse-ish graph
    x_cons = rng.normal(0.5, 0.03, n)
    G_cons = nx.erdos_renyi_graph(n, 0.06, seed=int(rng.integers(1e9)))

    # Low diversity, high tension: "narcissism of small differences"
    # still low-ish variance compared to polarized camps, but dense coupling amplifies friction.
    x_fric = rng.normal(0.0, 0.22, n)   # variance ~0.05
    G_fric = nx.complete_graph(n)

    # High diversity, low tension: echo chambers (two camps, almost disconnected)
    x_echo = make_two_camp_signal(n2, low=-1, high=1, seed=int(rng.integers(1e9)))
    G_echo = nx.planted_partition_graph(2, n2, p_in=0.18, p_out=0.0005, seed=int(rng.integers(1e9)))

    # High diversity, high tension: integrated conflict (two camps well mixed)
    x_conf = make_two_camp_signal(n2, low=-1, high=1, seed=int(rng.integers(1e9)))
    G_conf = nx.planted_partition_graph(2, n2, p_in=0.12, p_out=0.12, seed=int(rng.integers(1e9)))

    scenarios = [
        ("Consensus (aligned community)", G_cons, x_cons),
        ("Friction (small diffs, dense ties)", G_fric, x_fric),
        ("Echo chambers (segregation)", G_echo, x_echo),
        ("Integrated conflict (forced mixing)", G_conf, x_conf),
    ]

    rows = []
    for name, G, x in scenarios:
        m = get_metrics_full(G, x)
        rows.append({
            "scenario": name,
            "diversity_var": m["var"],
            "diversity_gini_signed": m["gini_signed"],
            "tension_rq_l2": m["rq_l2"],
            "tension_de_per_edge": m["de_per_edge"],
            "edges": int(m["m"]),
        })
    return pd.DataFrame(rows)


def topology_sweep_same_opinions(seed: int = 42, n_per: int = 50, p_in: float = 0.35, p_out_grid: np.ndarray = None) -> pd.DataFrame:
    """
    Keep the SAME two-camp opinions x in {-1,+1}. Vary topology mixing p_out.
    For this x, Dirichlet energy is exactly:
        DE = sum_{(i,j)} (x_i - x_j)^2 = 4 * (# cross edges)
    so tension is a direct proxy for "how much the two camps are forced to interact".
    """
    if p_out_grid is None:
        p_out_grid = np.linspace(0.0, 0.25, 11)

    rng = np.random.default_rng(seed)
    x = make_two_camp_signal(n_per, low=-1, high=1, seed=int(rng.integers(1e9)))

    rows = []
    for p_out in p_out_grid:
        G = nx.planted_partition_graph(2, n_per, p_in=p_in, p_out=float(p_out), seed=int(rng.integers(1e9)))
        m = get_metrics_full(G, x)
        rows.append({
            "p_out": float(p_out),
            "edges": int(m["m"]),
            "de": m["de"],
            "rq_l2": m["rq_l2"],
            "var": m["var"],
        })
    return pd.DataFrame(rows)


def radicalization_fracture_mediation(
    seed: int = 42,
    n: int = 100,
    radical_strength: float = 2.0,
    frac_remove: float = 0.9,
    n_mediators: int = 20,
    mediator_opinion: float = 0.0,
) -> pd.DataFrame:
    """
    Signed space chain:
      1) consensus-ish
      2) radicalization (half shift +radical_strength)
      3) fracture (remove cross edges)
      4) mediation (add bridge nodes at mediator_opinion)
    """
    rng = np.random.default_rng(seed)
    n2 = n // 2

    x1 = rng.normal(0.0, 0.1, n)
    G1 = nx.erdos_renyi_graph(n, 0.08, seed=int(rng.integers(1e9)))

    x2 = x1.copy()
    x2[:n2] += radical_strength
    G2 = G1

    G3 = G1.copy()
    cross_edges = [(u, v) for (u, v) in list(G3.edges())
                   if (u < n2 and v >= n2) or (u >= n2 and v < n2)]
    if cross_edges:
        k = int(frac_remove * len(cross_edges))
        idx = rng.choice(len(cross_edges), size=k, replace=False)
        G3.remove_edges_from([cross_edges[i] for i in idx])

    G4 = G3.copy()
    x4 = x2.tolist()
    start = n
    for i in range(n_mediators):
        node = start + i
        G4.add_node(node)
        x4.append(float(mediator_opinion))
        targets_A = rng.choice(n2, size=6, replace=False).tolist()
        targets_B = rng.choice(range(n2, n), size=6, replace=False).tolist()
        for t in targets_A + targets_B:
            G4.add_edge(node, t)
    x4 = np.array(x4, dtype=float)

    chain = [
        ("1. Consensus", G1, x1),
        ("2. Conflict (radicalization)", G2, x2),
        ("3. Fracture (echo)", G3, x2),
        (f"4. Mediation (bridges at {mediator_opinion:+.2f})", G4, x4),
    ]

    rows = []
    for label, G, x in chain:
        m = get_metrics_full(G, x)
        rows.append({
            "state": label,
            "var": m["var"],
            "rq_l2": m["rq_l2"],
            "gini_signed": m["gini_signed"],
            "de_per_edge": m["de_per_edge"],
            "edges": int(m["m"]),
            "nodes": int(m["n"]),
        })
    return pd.DataFrame(rows)


# =============================================================================
# 4) PLOTS
# =============================================================================

def plot_quadrant_map(df: pd.DataFrame, xcol: str, ycol: str, title: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.scatter(df[xcol], df[ycol])
    for _, row in df.iterrows():
        plt.annotate(row["scenario"], (row[xcol], row[ycol]), textcoords="offset points", xytext=(6, 6))
    plt.title(title)
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.axvline(df[xcol].median(), linestyle=":")
    plt.axhline(df[ycol].median(), linestyle=":")
    plt.tight_layout()
    plt.show()


def plot_transition(df_chain: pd.DataFrame, xcol: str, ycol: str, title: str) -> None:
    plt.figure(figsize=(9, 6))
    plt.plot(df_chain[xcol], df_chain[ycol], marker="o")
    for _, row in df_chain.iterrows():
        plt.annotate(row["state"], (row[xcol], row[ycol]), textcoords="offset points", xytext=(6, 6))
    plt.title(title)
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.tight_layout()
    plt.show()


def plot_dynamics_plane(deg: SimulationHistory, bc: SimulationHistory, xcol: str, ycol: str, title: str) -> None:
    df1 = pd.DataFrame(deg.metrics_hist)
    df2 = pd.DataFrame(bc.metrics_hist)

    plt.figure(figsize=(9, 6))
    plt.plot(df1[xcol], df1[ycol], label="DeGroot")
    plt.plot(df2[xcol], df2[ycol], label="Bounded-confidence", linestyle="--")
    plt.title(title)
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_group_means(deg: SimulationHistory, bc: SimulationHistory, split: int) -> None:
    t1 = np.arange(deg.x_hist.shape[0])
    t2 = np.arange(bc.x_hist.shape[0])
    dA, dB = group_means(deg.x_hist, split=split)
    bA, bB = group_means(bc.x_hist, split=split)

    plt.figure(figsize=(10, 4))
    plt.plot(t1, dA, label="DeGroot: mean(A)")
    plt.plot(t1, dB, label="DeGroot: mean(B)")
    plt.plot(t2, bA, linestyle="--", label="Bounded: mean(A)")
    plt.plot(t2, bB, linestyle="--", label="Bounded: mean(B)")
    plt.title("Macro polarization: camp means over time")
    plt.xlabel("Time (iterations)")
    plt.ylabel("Opinion")
    plt.legend()
    plt.tight_layout()
    plt.show()


def phase_diagram_mix_threshold(
    seed: int = 42,
    n_per: int = 35,
    p_in: float = 0.35,
    p_out_grid: np.ndarray = None,
    thr_grid: np.ndarray = None,
    init_noise: float = 0.12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sweep:
      - p_out: cross-group mixing in the topology
      - threshold: intolerance radius in bounded-confidence

    With strict camps +/-1, the transition is near threshold ~2.
    To avoid a degenerate diagram, we add small init_noise around the camps.

    Outputs matrices (len(thr_grid) x len(p_out_grid)):
      - polarization_final = |meanA - meanB|
      - rq_l2_final
      - gini_signed_final
      - var_final
      - de_per_edge_final
    """
    if p_out_grid is None:
        p_out_grid = np.linspace(0.0, 0.25, 9)
    if thr_grid is None:
        thr_grid = np.linspace(0.5, 2.6, 12)

    rng = np.random.default_rng(seed)
    split = n_per

    pol = np.zeros((len(thr_grid), len(p_out_grid)))
    rq = np.zeros_like(pol)
    gsig = np.zeros_like(pol)
    var = np.zeros_like(pol)
    depe = np.zeros_like(pol)

    x0 = make_two_camp_signal(n_per, low=-1, high=1, noise=init_noise, seed=int(rng.integers(1e9)))

    for j, p_out in enumerate(p_out_grid):
        G = nx.planted_partition_graph(2, n_per, p_in=p_in, p_out=float(p_out), seed=int(rng.integers(1e9)))
        for i, thr in enumerate(thr_grid):
            hist = simulate_bounded_confidence(G, x0, threshold=float(thr), steps=100)
            xT = hist.x_hist[-1]
            a = float(np.mean(xT[:split]))
            b = float(np.mean(xT[split:]))
            pol[i, j] = abs(a - b)

            mT = hist.metrics_hist[-1]
            rq[i, j] = mT["rq_l2"]
            gsig[i, j] = mT["gini_signed"]
            var[i, j] = mT["var"]
            depe[i, j] = mT["de_per_edge"]

    return pol, rq, gsig, var, depe, p_out_grid, thr_grid


def plot_heatmap(mat: np.ndarray, x_ticks: List[float], y_ticks: List[float], title: str, xlabel: str, ylabel: str) -> None:
    plt.figure(figsize=(10, 4.8))
    plt.imshow(mat, aspect="auto", origin="lower")
    plt.xticks(ticks=np.arange(len(x_ticks)), labels=[f"{v:.2f}" for v in x_ticks], rotation=45)
    plt.yticks(ticks=np.arange(len(y_ticks)), labels=[f"{v:.2f}" for v in y_ticks])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar()
    plt.tight_layout()
    plt.show()


# =============================================================================
# 5) MAIN
# =============================================================================

def run_all(seed: int = 42, show: bool = True) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}

    # --- Quadrant archetypes
    df_quad = quadrant_archetypes(seed=seed, n=100)
    out["quadrants"] = df_quad
    print("\n=== Quadrant archetypes ===")
    print(df_quad.to_string(index=False))

    if show:
        plot_quadrant_map(
            df_quad,
            xcol="diversity_var",
            ycol="tension_rq_l2",
            title="Quadrant map (toy): Diversity (var) vs Tension (RQ_L2)"
        )

    # --- Topology sweep: same opinions, varying mixing
    df_sweep = topology_sweep_same_opinions(seed=seed, n_per=50)
    out["topology_sweep"] = df_sweep
    print("\n=== Topology sweep (same two-camp opinions) ===")
    print(df_sweep.to_string(index=False))

    if show:
        plt.figure(figsize=(9, 4))
        plt.plot(df_sweep["p_out"], df_sweep["rq_l2"], marker="o")
        plt.title("Topology sweep: RQ_L2 increases with cross-group mixing (same opinions)")
        plt.xlabel("p_out (cross-group mixing)")
        plt.ylabel("RQ_L2")
        plt.tight_layout()
        plt.show()

    # --- Transition chain: neutral vs biased mediators
    df_chain_neu = radicalization_fracture_mediation(seed=seed + 1, mediator_opinion=0.0)
    df_chain_bias = radicalization_fracture_mediation(seed=seed + 2, mediator_opinion=0.4)
    out["transition_neutral"] = df_chain_neu
    out["transition_biased"] = df_chain_bias

    print("\n=== Transition: neutral mediators ===")
    print(df_chain_neu.to_string(index=False))
    print("\n=== Transition: biased mediators (+0.4) ===")
    print(df_chain_bias.to_string(index=False))

    if show:
        plot_transition(df_chain_neu, xcol="var", ycol="rq_l2", title="Transition in (var, RQ_L2): neutral mediators")
        plot_transition(df_chain_bias, xcol="var", ycol="rq_l2", title="Transition in (var, RQ_L2): biased mediators")

    # --- Dynamics trajectory in the quadrant plane
    n_per = 50
    G_mix = nx.planted_partition_graph(2, n_per, p_in=0.35, p_out=0.12, seed=seed)
    x0 = make_two_camp_signal(n_per, low=-1, high=1, seed=seed)

    deg = simulate_degroot(G_mix, x0, steps=35)
    bc = simulate_bounded_confidence(G_mix, x0, threshold=0.35, steps=80)

    if show:
        plot_dynamics_plane(deg, bc, xcol="var", ycol="rq_l2", title="Dynamics plane: DeGroot vs Bounded-confidence")
        plot_group_means(deg, bc, split=n_per)

    # --- Phase diagrams
    pol, rq, gsig, var, depe, p_out_grid, thr_grid = phase_diagram_mix_threshold(seed=seed, n_per=35)
    if show:
        plot_heatmap(pol, list(p_out_grid), list(thr_grid), "Phase diagram: final polarization |meanA-meanB|", "p_out", "threshold")
        plot_heatmap(rq, list(p_out_grid), list(thr_grid), "Phase diagram: final tension RQ_L2", "p_out", "threshold")

    out["phase_polarization"] = pd.DataFrame(pol, index=thr_grid, columns=p_out_grid)
    out["phase_rq_l2"] = pd.DataFrame(rq, index=thr_grid, columns=p_out_grid)

    return out


if __name__ == "__main__":
    run_all(seed=42, show=True)
