"""
gdrq: Gini / Dirichlet / Rayleigh Quotient utilities on graphs.

This package provides:
- Graph utilities (adjacency validation, Laplacians)
- Signal utilities (centering, norms)
- Metrics (Gini-type pairwise dispersion, graph TV, Dirichlet, RQ)
- Cauchy-Schwarz bounds linking L1 and L2 graph dispersions
- Cut/conductance helpers
- Spectral helpers (Fiedler vector)
- Basic CSBM generator and experiments
"""

from .graph_utils import (
    ArrayLike,
    validate_adjacency,
    degree_vector,
    degree_matrix,
    laplacian,
    normalized_laplacian,
    complete_graph_adjacency,
    edge_mass,
    number_of_pairs,
)

from .signal_utils import (
    as_signal,
    center_signal,
    euclidean_norm_sq,
    weighted_norm_sq,
)

from .metrics import (
    pairwise_differences,
    gini_pairwise_numerator,
    weighted_total_variation,
    dirichlet_quadratic,
    dirichlet_half_energy,
    dirichlet_pairwise_ordered_halfsum,
    rayleigh_quotient,
    normalized_rayleigh_quotient,
    complete_graph_dirichlet_identity_terms,
    centered_complete_graph_constant_check,
    cauchy_schwarz_bound_complete,
    cauchy_schwarz_bound_complete_centered_to_norm,
    cauchy_schwarz_bound_weighted,
    dirichlet_gradient,
    tv_subgradient,
)

from .cuts import (
    cut_value,
    volume,
    conductance,
    sweep_cut_from_scores,
)

from .spectral import (
    fiedler_vector,
)

from .csbm import (
    CSBMData,
    generate_csbm,
)

from .experiments import (
    summarize_signal_metrics,
    run_basic_csbm_experiment,
    run_csbm_grid_experiments,
    demo_small_sanity_checks,
    demo_csbm,
    demo_csbm_grid,
)

from .plots import (
    plot_tv_vs_dirichlet,
    plot_signal_metric_bars_from_basic_experiment,
    plot_grid_metric_vs_mu,
    plot_grid_metric_vs_pout,
    plot_grid_conductance_comparison,
    run_and_plot_grid_metric_vs_mu,
    run_and_plot_grid_metric_vs_pout,
    run_and_plot_basic_experiment_bars,
    demo_plots,
)

__all__ = [
    # graph_utils
    "ArrayLike",
    "validate_adjacency",
    "degree_vector",
    "degree_matrix",
    "laplacian",
    "normalized_laplacian",
    "complete_graph_adjacency",
    "edge_mass",
    "number_of_pairs",
    # signal_utils
    "as_signal",
    "center_signal",
    "euclidean_norm_sq",
    "weighted_norm_sq",
    # metrics
    "pairwise_differences",
    "gini_pairwise_numerator",
    "weighted_total_variation",
    "dirichlet_quadratic",
    "dirichlet_half_energy",
    "dirichlet_pairwise_ordered_halfsum",
    "rayleigh_quotient",
    "normalized_rayleigh_quotient",
    "complete_graph_dirichlet_identity_terms",
    "centered_complete_graph_constant_check",
    "cauchy_schwarz_bound_complete",
    "cauchy_schwarz_bound_complete_centered_to_norm",
    "cauchy_schwarz_bound_weighted",
    "dirichlet_gradient",
    "tv_subgradient",
    # cuts
    "cut_value",
    "volume",
    "conductance",
    "sweep_cut_from_scores",
    # spectral
    "fiedler_vector",
    # csbm
    "CSBMData",
    "generate_csbm",
    # experiments
    "summarize_signal_metrics",
    "run_basic_csbm_experiment",
    "run_csbm_grid_experiments",
    "demo_small_sanity_checks",
    "demo_csbm",
    "demo_csbm_grid",
    # plots
    "plot_tv_vs_dirichlet",
    "plot_signal_metric_bars_from_basic_experiment",
    "plot_grid_metric_vs_mu",
    "plot_grid_metric_vs_pout",
    "plot_grid_conductance_comparison",
    "run_and_plot_grid_metric_vs_mu",
    "run_and_plot_grid_metric_vs_pout",
    "run_and_plot_basic_experiment_bars",
    "demo_plots",
]