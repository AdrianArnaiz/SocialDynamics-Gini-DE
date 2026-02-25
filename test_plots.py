import gdrq_1 as gg

# Generate grid rows
rows = gg.run_csbm_grid_experiments(seed=0)

# Plot correlations vs mu
gg.plot_grid_metric_vs_mu(rows, y_key="feature_label_corr")
gg.plot_grid_metric_vs_mu(rows, y_key="fiedler_label_corr_abs")

# Plot conductance comparison vs p_out at fixed mu=1.0
gg.plot_grid_conductance_comparison(rows, x_axis="p_out", fixed={"mu": 1.0})

# Run one experiment and compare signal metrics
res = gg.run_basic_csbm_experiment(seed=0)
gg.plot_signal_metric_bars_from_basic_experiment(res, metric_key="dirichlet_Q")
gg.plot_signal_metric_bars_from_basic_experiment(res, metric_key="tv_weighted")