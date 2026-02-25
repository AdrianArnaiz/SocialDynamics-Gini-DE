from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

from .experiments import run_basic_csbm_experiment, run_csbm_grid_experiments


Row = Dict[str, float]


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------

def _rows_to_list(rows: Union[List[Row], Sequence[Row]]) -> List[Row]:
    return list(rows)


def _sorted_unique(values: Iterable[float]) -> List[float]:
    vals = sorted({float(v) for v in values})
    return vals


def _filter_rows(rows: Sequence[Row], **fixed_params: float) -> List[Row]:
    """
    Filter rows by exact numeric matches on keys.
    """
    out: List[Row] = []
    for r in rows:
        keep = True
        for k, v in fixed_params.items():
            if k not in r or float(r[k]) != float(v):
                keep = False
                break
        if keep:
            out.append(r)
    return out


def _group_rows(rows: Sequence[Row], key: str) -> Dict[float, List[Row]]:
    groups: Dict[float, List[Row]] = {}
    for r in rows:
        kval = float(r[key])
        groups.setdefault(kval, []).append(r)
    return groups


def _sort_rows_by(rows: Sequence[Row], key: str) -> List[Row]:
    return sorted(rows, key=lambda r: float(r[key]))


def _maybe_savefig(savepath: Optional[str], dpi: int = 140, tight: bool = True) -> None:
    if tight:
        plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=dpi, bbox_inches="tight")


# ---------------------------------------------------------------------
# Metric diagnostics plots
# ---------------------------------------------------------------------

def plot_tv_vs_dirichlet(
    rows: Sequence[Row],
    *,
    signal: str = "features",
    loglog: bool = False,
    annotate: bool = False,
    savepath: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Scatter plot of TV vs Dirichlet over a grid of experiments.

    Parameters
    ----------
    rows : sequence of dict
        Output rows from run_csbm_grid_experiments.
    signal : {"labels", "features", "fiedler"}
        Which signal to plot.
    loglog : bool
        Use log-log axes.
    annotate : bool
        Annotate points with (p_out, mu).
    savepath : str or None
        If given, save figure to this path.
    show : bool
        If True, call plt.show().

    Notes
    -----
    Expects row keys:
      TV_<signal>, Q_<signal>, p_out, mu
    """
    rows = _rows_to_list(rows)
    tv_key = f"TV_{signal}"
    q_key = f"Q_{signal}"

    if not rows:
        raise ValueError("rows is empty.")
    if tv_key not in rows[0] or q_key not in rows[0]:
        raise ValueError(f"Missing keys '{tv_key}'/'{q_key}' in rows.")

    x = np.array([float(r[tv_key]) for r in rows], dtype=float)
    y = np.array([float(r[q_key]) for r in rows], dtype=float)

    plt.figure(figsize=(7, 5))
    plt.scatter(x, y)
    plt.xlabel(f"{tv_key}")
    plt.ylabel(f"{q_key}")
    plt.title(f"TV vs Dirichlet for '{signal}' signal across CSBM grid")

    if loglog:
        # avoid invalid log scales if nonpositive values appear
        if np.any(x <= 0) or np.any(y <= 0):
            raise ValueError("loglog=True requires all plotted values to be positive.")
        plt.xscale("log")
        plt.yscale("log")

    if annotate:
        for xi, yi, r in zip(x, y, rows):
            txt = f"(p_out={r.get('p_out')}, mu={r.get('mu')})"
            plt.annotate(txt, (xi, yi), fontsize=8)

    _maybe_savefig(savepath)
    if show:
        plt.show()
    else:
        plt.close()


def plot_signal_metric_bars_from_basic_experiment(
    result: Dict[str, object],
    *,
    metric_key: str = "dirichlet_Q",
    savepath: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Bar plot comparing one metric across:
      labels_centered, features_centered, fiedler_centered

    Parameters
    ----------
    result : dict
        Output of run_basic_csbm_experiment(...)
    metric_key : str
        One of the keys inside result["metrics"][...], e.g.
        'tv_weighted', 'dirichlet_Q', 'rayleigh', 'rayleigh_normalized'
    """
    metrics = result["metrics"]  # type: ignore[index]
    keys = ["labels_centered", "features_centered", "fiedler_centered"]
    labels = ["labels", "features", "fiedler"]
    vals = []

    for k in keys:
        if metric_key not in metrics[k]:  # type: ignore[index]
            raise ValueError(f"Metric '{metric_key}' not found in result['metrics']['{k}'].")
        vals.append(float(metrics[k][metric_key]))  # type: ignore[index]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, vals)
    plt.ylabel(metric_key)
    plt.title(f"Signal comparison for {metric_key}")

    _maybe_savefig(savepath)
    if show:
        plt.show()
    else:
        plt.close()


# ---------------------------------------------------------------------
# CSBM grid trend plots
# ---------------------------------------------------------------------

def plot_grid_metric_vs_mu(
    rows: Sequence[Row],
    *,
    y_key: str = "fiedler_label_corr_abs",
    p_out_values: Optional[Sequence[float]] = None,
    savepath: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot a chosen metric vs mu, with one line per p_out.

    Parameters
    ----------
    rows : sequence of dict
        Output rows from run_csbm_grid_experiments.
    y_key : str
        Metric key in each row, e.g.
        'fiedler_label_corr_abs', 'feature_label_corr',
        'true_conductance', 'sweep_fiedler_conductance', ...
    p_out_values : sequence or None
        Restrict to these p_out values (all if None).
    """
    rows = _rows_to_list(rows)
    if not rows:
        raise ValueError("rows is empty.")
    if y_key not in rows[0]:
        raise ValueError(f"Key '{y_key}' not found in rows.")

    if p_out_values is None:
        p_out_values = _sorted_unique(r["p_out"] for r in rows)

    plt.figure(figsize=(7.5, 5.2))

    for p_out in p_out_values:
        sub = _filter_rows(rows, p_out=float(p_out))
        sub = _sort_rows_by(sub, "mu")
        if not sub:
            continue
        x = [float(r["mu"]) for r in sub]
        y = [float(r[y_key]) for r in sub]
        plt.plot(x, y, marker="o", label=f"p_out={p_out:g}")

    plt.xlabel("mu (context signal strength)")
    plt.ylabel(y_key)
    plt.title(f"{y_key} vs mu across p_out")
    plt.legend()

    _maybe_savefig(savepath)
    if show:
        plt.show()
    else:
        plt.close()


def plot_grid_metric_vs_pout(
    rows: Sequence[Row],
    *,
    y_key: str = "sweep_fiedler_conductance",
    mu_values: Optional[Sequence[float]] = None,
    savepath: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot a chosen metric vs p_out, with one line per mu.
    """
    rows = _rows_to_list(rows)
    if not rows:
        raise ValueError("rows is empty.")
    if y_key not in rows[0]:
        raise ValueError(f"Key '{y_key}' not found in rows.")

    if mu_values is None:
        mu_values = _sorted_unique(r["mu"] for r in rows)

    plt.figure(figsize=(7.5, 5.2))

    for mu in mu_values:
        sub = _filter_rows(rows, mu=float(mu))
        sub = _sort_rows_by(sub, "p_out")
        if not sub:
            continue
        x = [float(r["p_out"]) for r in sub]
        y = [float(r[y_key]) for r in sub]
        plt.plot(x, y, marker="o", label=f"mu={mu:g}")

    plt.xlabel("p_out (inter-block edge probability)")
    plt.ylabel(y_key)
    plt.title(f"{y_key} vs p_out across mu")
    plt.legend()

    _maybe_savefig(savepath)
    if show:
        plt.show()
    else:
        plt.close()


def plot_grid_conductance_comparison(
    rows: Sequence[Row],
    *,
    x_axis: str = "p_out",
    fixed: Optional[Dict[str, float]] = None,
    savepath: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Compare conductance curves:
      - true_conductance
      - sweep_feature_conductance
      - sweep_fiedler_conductance

    Parameters
    ----------
    x_axis : {"p_out", "mu"}
        Which variable to place on x-axis.
    fixed : dict or None
        Fix the other variable(s), e.g. {"mu": 1.0} if x_axis="p_out".
    """
    rows = _rows_to_list(rows)
    if x_axis not in ("p_out", "mu"):
        raise ValueError("x_axis must be 'p_out' or 'mu'.")

    if fixed:
        rows = _filter_rows(rows, **fixed)
    if not rows:
        raise ValueError("No rows left after filtering.")

    rows = _sort_rows_by(rows, x_axis)
    x = [float(r[x_axis]) for r in rows]

    plt.figure(figsize=(7.5, 5.2))
    plt.plot(x, [float(r["true_conductance"]) for r in rows], marker="o", label="true_conductance")
    plt.plot(x, [float(r["sweep_feature_conductance"]) for r in rows], marker="o", label="sweep_feature_conductance")
    plt.plot(x, [float(r["sweep_fiedler_conductance"]) for r in rows], marker="o", label="sweep_fiedler_conductance")

    title = f"Conductance comparison vs {x_axis}"
    if fixed:
        fixed_txt = ", ".join(f"{k}={v:g}" for k, v in fixed.items())
        title += f" ({fixed_txt})"

    plt.xlabel(x_axis)
    plt.ylabel("conductance")
    plt.title(title)
    plt.legend()

    _maybe_savefig(savepath)
    if show:
        plt.show()
    else:
        plt.close()


# ---------------------------------------------------------------------
# High-level convenience runners (generate rows + plot)
# ---------------------------------------------------------------------

def run_and_plot_grid_metric_vs_mu(
    *,
    y_key: str = "fiedler_label_corr_abs",
    n: int = 200,
    p_in: float = 0.15,
    p_out_list: Sequence[float] = (0.01, 0.03, 0.05, 0.08),
    mu_list: Sequence[float] = (0.2, 0.5, 1.0, 2.0),
    sigma: float = 1.0,
    seed: int = 0,
    savepath: Optional[str] = None,
    show: bool = True,
) -> List[Row]:
    """
    Run a CSBM grid and immediately plot y_key vs mu.
    Returns the rows used for plotting.
    """
    rows = run_csbm_grid_experiments(
        n=n,
        p_in=p_in,
        p_out_list=p_out_list,
        mu_list=mu_list,
        sigma=sigma,
        seed=seed,
    )
    plot_grid_metric_vs_mu(rows, y_key=y_key, savepath=savepath, show=show)
    return rows


def run_and_plot_grid_metric_vs_pout(
    *,
    y_key: str = "sweep_fiedler_conductance",
    n: int = 200,
    p_in: float = 0.15,
    p_out_list: Sequence[float] = (0.01, 0.03, 0.05, 0.08),
    mu_list: Sequence[float] = (0.2, 0.5, 1.0, 2.0),
    sigma: float = 1.0,
    seed: int = 0,
    savepath: Optional[str] = None,
    show: bool = True,
) -> List[Row]:
    """
    Run a CSBM grid and immediately plot y_key vs p_out.
    Returns the rows used for plotting.
    """
    rows = run_csbm_grid_experiments(
        n=n,
        p_in=p_in,
        p_out_list=p_out_list,
        mu_list=mu_list,
        sigma=sigma,
        seed=seed,
    )
    plot_grid_metric_vs_pout(rows, y_key=y_key, savepath=savepath, show=show)
    return rows


def run_and_plot_basic_experiment_bars(
    *,
    metric_key: str = "dirichlet_Q",
    n: int = 200,
    p_in: float = 0.15,
    p_out: float = 0.03,
    mu: float = 1.0,
    sigma: float = 1.0,
    seed: int = 0,
    use_normalized_fiedler: bool = True,
    savepath: Optional[str] = None,
    show: bool = True,
) -> Dict[str, object]:
    """
    Run one CSBM experiment and plot a bar comparison for a selected metric.
    Returns the experiment result dict.
    """
    res = run_basic_csbm_experiment(
        n=n,
        p_in=p_in,
        p_out=p_out,
        mu=mu,
        sigma=sigma,
        seed=seed,
        use_normalized_fiedler=use_normalized_fiedler,
    )
    plot_signal_metric_bars_from_basic_experiment(res, metric_key=metric_key, savepath=savepath, show=show)
    return res


# ---------------------------------------------------------------------
# Demo entrypoint
# ---------------------------------------------------------------------

def demo_plots(seed: int = 0) -> None:
    """
    Minimal plotting demo for the package.
    """
    rows = run_csbm_grid_experiments(seed=seed)

    # 1) TV vs Dirichlet for features
    plot_tv_vs_dirichlet(rows, signal="features", loglog=False, annotate=False)

    # 2) Correlation trends vs mu
    plot_grid_metric_vs_mu(rows, y_key="feature_label_corr")
    plot_grid_metric_vs_mu(rows, y_key="fiedler_label_corr_abs")

    # 3) Conductance comparison vs p_out at fixed mu=1.0 (if present)
    mu_values = _sorted_unique(r["mu"] for r in rows)
    if 1.0 in mu_values:
        plot_grid_conductance_comparison(rows, x_axis="p_out", fixed={"mu": 1.0})
    else:
        # fallback to first available mu
        plot_grid_conductance_comparison(rows, x_axis="p_out", fixed={"mu": mu_values[0]})

    # 4) Bar comparison for one basic experiment
    res = run_basic_csbm_experiment(seed=seed)
    plot_signal_metric_bars_from_basic_experiment(res, metric_key="dirichlet_Q")
    plot_signal_metric_bars_from_basic_experiment(res, metric_key="tv_weighted")


if __name__ == "__main__":
    demo_plots(seed=0)