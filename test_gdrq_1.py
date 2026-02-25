import gdrq_1 as gg  # Package under test

# Small random example
import numpy as np
rng = np.random.default_rng(0)  # Deterministic RNG for reproducibility
use_csbm_graph = True  # Toggle between random graph and CSBM graph

if use_csbm_graph:
	data = gg.generate_csbm(n=10, p_in=0.5, p_out=0.1, mu=1.0, sigma=1.0, seed=0)
	W = data.W  # CSBM adjacency
	x = data.features  # CSBM contextual signal
else:
	W = rng.random((10, 10))  # Dense random adjacency weights
	W = 0.5 * (W + W.T)  # Symmetrize to make an undirected graph
	np.fill_diagonal(W, 0)  # Remove self-loops
	x = rng.normal(size=10)  # Random node signal

print("TV:", gg.weighted_total_variation(x, W))  # Graph total variation
print("Dirichlet:", gg.dirichlet_quadratic(x, W))  # Dirichlet energy
print("RQ:", gg.rayleigh_quotient(x, W))  # Rayleigh quotient
print("CS bound:", gg.cauchy_schwarz_bound_weighted(x, W))  # CS upper bound

# CSBM demo
res = gg.run_basic_csbm_experiment(seed=0)  # Run a simple CSBM experiment
print(res["spectral"])  # Report spectral summary