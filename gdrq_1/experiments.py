from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .graph_utils import ArrayLike, validate_adjacency
from .signal_utils import as_signal, center_signal, euclidean_norm_sq, weighted_norm_sq
from .metrics import (
    weighted_total_variation,
    dirichlet_quadratic,
    rayleigh_quotient,
    cauchy_schwarz_bound_weighted,
    gini_pairwise_numerator,
    complete_graph_dirichlet_identity_terms,
    cauchy_schwarz_bound_complete,
)
from .cuts import cut_value, conductance, sweep_cut_from_scores
from .spectral import fiedler_vector
from .csbm import generate_csbm


def summarize_signal_metrics(x: ArrayLike, W: ArrayLike, *, center_for_complete_check: bool = False) -> Dict[str, float]:
    """
    Compute a bundle of metrics for a signal x on graph W.
    """
    W = validate_adjacency(W)
    x = as_signal(x, n=W.shape[0])

    x_used = center_signal(x) if center_for_complete_check else x

    out = {
        "n": float(len(x)),
        "mean_x": float(x.mean()),
        "norm2": euclidean_norm_sq(x),
        "tv_weighted": weighted_total_variation(x, W),
        "dirichlet_Q": dirichlet_quadratic(x, W),
        "rayleigh": rayleigh_quotient(x, W) if euclidean_norm_sq(x) > 1e-15 else float("nan"),
    }

    dnorm = weighted_norm_sq(x, W)
    out["weighted_norm2_D"] = dnorm
    out["rayleigh_normalized"] = dirichlet_quadratic(x, W) / dnorm if dnorm > 1e-15 else float("nan")

    # Weighted C-S bound
    cwb = cauchy_schwarz_bound_weighted(x, W)
    out["cs_bound_tv"] = cwb["bound_on_tv"]
    out["cs_bound_holds"] = cwb["holds"]

    # Complete-graph Gini (global disparity), mainly diagnostic
    out["gini_num_complete_unordered"] = gini_pairwise_numerator(x, ordered=False)

    # Optional check on complete graph centered identity
    if center_for_complete_check:
        chk = complete_graph_dirichlet_identity_terms(x_used)
        out["complete_identity_abs_error"] = chk["abs_error_Q_rhs1"]

    return out


def run_basic_csbm_experiment(
    n: int = 200,
    p_in: float = 0.15,
    p_out: float = 0.03,
    mu: float = 1.0,
    sigma: float = 1.0,
    *,
    seed: Optional[int] = 0,
    use_normalized_fiedler: bool = True,
) -> Dict[str, object]:
    """
    Run a very basic CSBM experiment and compare metrics for:
    - true labels y in {-1,+1}
    - contextual feature x
    - Fiedler vector v2 (spectral embedding)

    Returns a dictionary with data and metric summaries.
    """
    data = generate_csbm(n=n, p_in=p_in, p_out=p_out, mu=mu, sigma=sigma, seed=seed)
    W = data.W

    lam2, v2 = fiedler_vector(W, normalized=use_normalized_fiedler)

    # Center vectors for fair comparison of mean-free graph-signal dispersion
    y = center_signal(data.labels)
    x = center_signal(data.features)
    v = center_signal(v2)

    metrics = {
        "labels_centered": summarize_signal_metrics(y, W),
        "features_centered": summarize_signal_metrics(x, W),
        "fiedler_centered": summarize_signal_metrics(v, W),
    }

    # Sweep cut quality from feature and fiedler scores
    sweep_feature = sweep_cut_from_scores(x, W)
    sweep_fiedler = sweep_cut_from_scores(v, W)

    # Ground-truth cut (using one block as S)
    S_true = np.where(data.blocks == 1)[0]
    true_cut = cut_value(S_true, W)
    true_phi = conductance(S_true, W)

    # Correlations with labels (sign-insensitive for fiedler)
    y_raw = data.labels
    feat_corr = float(np.corrcoef(y_raw, data.features)[0, 1])
    fied_corr = float(np.corrcoef(y_raw, v2)[0, 1])
    fied_corr_abs = abs(fied_corr)

    return {
        "params": {
            "n": n,
            "p_in": p_in,
            "p_out": p_out,
            "mu": mu,
            "sigma": sigma,
            "seed": seed,
            "use_normalized_fiedler": use_normalized_fiedler,
        },
        "spectral": {
            "lambda2": float(lam2),
            "fiedler_label_corr": fied_corr,
            "fiedler_label_corr_abs": float(fied_corr_abs),
            "feature_label_corr": feat_corr,
        },
        "ground_truth_partition": {
            "cut": float(true_cut),
            "conductance": float(true_phi),
            "size_S": int(len(S_true)),
        },
        "sweep_feature": sweep_feature,
        "sweep_fiedler": sweep_fiedler,
        "metrics": metrics,
        "data": data,  # includes W, labels, features, blocks
    }


def run_csbm_grid_experiments(
    *,
    n: int = 200,
    p_in: float = 0.15,
    p_out_list: Sequence[float] = (0.01, 0.03, 0.05, 0.08),
    mu_list: Sequence[float] = (0.2, 0.5, 1.0, 2.0),
    sigma: float = 1.0,
    seed: int = 0,
) -> List[Dict[str, float]]:
    """
    Very basic grid of CSBM experiments to see how graph separability (p_out)
    and feature strength (mu) affect spectral/feature behavior.
    """
    rows: List[Dict[str, float]] = []
    trial_id = 0
    for p_out in p_out_list:
        for mu in mu_list:
            trial_seed = seed + trial_id
            res = run_basic_csbm_experiment(
                n=n, p_in=p_in, p_out=p_out, mu=mu, sigma=sigma, seed=trial_seed
            )
            rows.append({
                "trial_id": float(trial_id),
                "n": float(n),
                "p_in": float(p_in),
                "p_out": float(p_out),
                "mu": float(mu),
                "sigma": float(sigma),
                "lambda2": float(res["spectral"]["lambda2"]),
                "feature_label_corr": float(res["spectral"]["feature_label_corr"]),
                "fiedler_label_corr_abs": float(res["spectral"]["fiedler_label_corr_abs"]),
                "true_conductance": float(res["ground_truth_partition"]["conductance"]),
                "sweep_feature_conductance": float(res["sweep_feature"]["best_conductance"]),
                "sweep_fiedler_conductance": float(res["sweep_fiedler"]["best_conductance"]),
                "Q_labels": float(res["metrics"]["labels_centered"]["dirichlet_Q"]),
                "Q_features": float(res["metrics"]["features_centered"]["dirichlet_Q"]),
                "Q_fiedler": float(res["metrics"]["fiedler_centered"]["dirichlet_Q"]),
                "TV_labels": float(res["metrics"]["labels_centered"]["tv_weighted"]),
                "TV_features": float(res["metrics"]["features_centered"]["tv_weighted"]),
                "TV_fiedler": float(res["metrics"]["fiedler_centered"]["tv_weighted"]),
            })
            trial_id += 1
    return rows


# ---------------------------------------------------------------------
# Demo helpers (printed examples)
# ---------------------------------------------------------------------

def _format_metric_block(name: str, d: Dict[str, float], keys: Sequence[str]) -> str:
    lines = [f"{name}:"]
    for k in keys:
        if k in d:
            val = d[k]
            if isinstance(val, (float, np.floating)):
                lines.append(f"  - {k}: {val:.6g}")
            else:
                lines.append(f"  - {k}: {val}")
    return "\n".join(lines)


def demo_small_sanity_checks(seed: int = 0) -> None:
    """
    Small deterministic-ish checks of identities/bounds.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=8)

    print("=== Small sanity checks ===")
    print("x:", np.round(x, 4))

    # Complete graph identity
    chk = complete_graph_dirichlet_identity_terms(x)
    print(_format_metric_block(
        "Complete graph identity",
        chk,
        ["Q_complete", "n_times_norm2_minus_sumx2", "n_times_centered_sumsq", "abs_error_Q_rhs1", "abs_error_Q_rhs2"]
    ))

    # Complete graph C-S bound
    cs = cauchy_schwarz_bound_complete(x)
    print(_format_metric_block(
        "Cauchy-Schwarz (complete graph)",
        cs,
        ["gini_num_unordered", "dirichlet_Q_complete", "lhs_gini_sq", "rhs_bound_sq", "slack_rhs_minus_lhs", "bound_on_gini"]
    ))

    # Weighted graph random symmetric matrix (small)
    n = 8
    A = rng.uniform(size=(n, n))
    W = (A + A.T) / 2.0
    np.fill_diagonal(W, 0.0)
    csw = cauchy_schwarz_bound_weighted(x, W, half_energy_convention=True)
    print(_format_metric_block(
        "Cauchy-Schwarz (weighted)",
        csw,
        ["tv_weighted", "dirichlet_Q", "edge_mass", "lhs_tv_sq", "rhs_bound_sq", "slack_rhs_minus_lhs", "bound_on_tv"]
    ))
    print()


def demo_csbm(seed: int = 0) -> None:
    """
    Run a basic CSBM experiment and print compact results.
    """
    print("=== CSBM demo ===")
    res = run_basic_csbm_experiment(
        n=200, p_in=0.12, p_out=0.03, mu=0.8, sigma=1.0, seed=seed, use_normalized_fiedler=True
    )

    print("Params:", res["params"])
    print("Spectral:", {k: round(v, 6) if isinstance(v, float) else v for k, v in res["spectral"].items()})
    print("Ground truth partition:", {k: round(v, 6) if isinstance(v, float) else v for k, v in res["ground_truth_partition"].items()})

    for key in ("labels_centered", "features_centered", "fiedler_centered"):
        print(_format_metric_block(
            key,
            res["metrics"][key],
            ["mean_x", "norm2", "tv_weighted", "dirichlet_Q", "rayleigh", "rayleigh_normalized", "cs_bound_tv"]
        ))

    print("Sweep feature best conductance:", res["sweep_feature"]["best_conductance"])
    print("Sweep fiedler best conductance:", res["sweep_fiedler"]["best_conductance"])
    print()


def demo_csbm_grid(seed: int = 0) -> None:
    """
    Run a small grid and print a few rows (no pandas dependency).
    """
    print("=== CSBM grid demo ===")
    rows = run_csbm_grid_experiments(seed=seed)
    print(f"Num runs: {len(rows)}")
    keys = [
        "p_out", "mu", "lambda2", "feature_label_corr", "fiedler_label_corr_abs",
        "true_conductance", "sweep_feature_conductance", "sweep_fiedler_conductance"
    ]
    for row in rows[:8]:
        print({k: round(row[k], 6) for k in keys})
    print()


if __name__ == "__main__":
    demo_small_sanity_checks(seed=0)
    demo_csbm(seed=1)
    demo_csbm_grid(seed=2)