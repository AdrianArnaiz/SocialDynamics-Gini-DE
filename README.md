# Social Dynamics: Gini Index & Dirichlet Energy on Graphs

## System Purpose

This toolkit bridges **inequality theory** and **graph spectral analysis** to diagnose social network structures. 

**Core Insight:** The Gini Index (a global $L_1$-dispersion measure) and Dirichlet Energy (a local $L_2$-smoothness measure) capture **orthogonal dimensions** of networked populations:
- **Gini Index**: How opinions/resources are distributed across people (composition)
- **Dirichlet Energy**: How those values align across social ties (structural tension)
- **Rayleigh Quotient**: Normalized roughness, revealing the **efficiency** of structure-signal coupling

**Master Map of Social States:** This framework identifies **four archetypal network states**:
1. **Consensus** (Low diversity, low tension) — Stable but stagnant
2. **Conflict** (High diversity, high tension) — Pre-revolutionary, toxic
3. **Fracture** (High diversity, low tension) — Echo chambers, cold polarization
4. **Mediation** (Medium diversity, moderate tension, optimal RQ) — Diverse yet cohesive (via bridge nodes)

**Why This Matters:** Standard network analysis tools either focus on topology (modularity, centrality) or treat node attributes as independent features. This framework **couples structure and signal**, enabling:
- Detection of structural fracture masked by low RQ (distinguishing Fracture from Consensus)
- Design of stable diverse networks via bridge-node mediation
- Prediction of network instability and transition paths
- Homophily vs. heterophily diagnostics with normalization

---

## Technical Anatomy

### Core Package: `gdrq_1/`

Production-ready library for graph signal analysis.

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| [`graph_utils.py`](gdrq_1/graph_utils.py) | Graph primitives | `laplacian()`, `normalized_laplacian()`, `validate_adjacency()`, `complete_graph_adjacency()` |
| [`signal_utils.py`](gdrq_1/signal_utils.py) | Signal preprocessing | `center_signal()`, `euclidean_norm_sq()`, `weighted_norm_sq()` |
| [`metrics.py`](gdrq_1/metrics.py) | Core disparities | `gini_pairwise_numerator()`, `dirichlet_quadratic()`, `rayleigh_quotient()`, `weighted_total_variation()` |
| [`spectral.py`](gdrq_1/spectral.py) | Eigenanalysis | `fiedler_vector()` (algebraic connectivity, spectral clustering) |
| [`csbm.py`](gdrq_1/csbm.py) | Synthetic data | `generate_csbm()` (Contextual Stochastic Block Model) |
| [`cuts.py`](gdrq_1/cuts.py) | Partitioning | `cut_value()`, `conductance()`, `sweep_cut_from_scores()` |
| [`experiments.py`](gdrq_1/experiments.py) | Reproducible workflows | `run_basic_csbm_experiment()`, `summarize_signal_metrics()` |

### Documentation: `doc/`

Mathematical foundations and proofs.

| Document | Maps To | Content |
|----------|---------|---------|
| [gini_and_rq_gpt.md](doc/gini_and_rq_gpt.md) | `metrics.py` | Formal correspondence: Gini ↔ Dirichlet on complete graphs, $L_1/L_2$ inequalities, spectral gap dynamics |
| [gini_and_rq_1.md](doc/gini_and_rq_1.md) | `metrics.py` | Pairwise-sum conventions, exact identities on $K_n$, Rayleigh quotient interpretation |
| [use_case_sn.md](doc/use_case_sn.md) | `use_case_sn.py` | **Social networks**: echo chambers vs. integration, homophily detection, Rayleigh quotient as consensus predictor |
| [gram_and_rq.md](doc/gram_and_rq.md) | `graph_utils.py` | Gram matrix ($X^T X$), $L_2$ norms, Frobenius norm, duality of Rayleigh denominators |
| [variance_spectral.md](doc/variance_spectral.md) | `signal_utils.py` | Variance identities: pairwise form, spectral decay under heat diffusion |
| [variance.md](doc/variance.md) | `benchmark_variance.py` | Computational trade-offs: NumPy vs. pairwise vs. dot-product variance |

### Experiment Scripts

Applied use cases demonstrating the framework.

| Script | Purpose |
|--------|---------|
| [use_case_sn.py](use_case_sn.py) | **Master Map scenarios**: Opinion dynamics (DeGroot vs. bounded-confidence), 4-state transitions (Consensus → Conflict → Fracture → Mediation), bridge-node stabilization |
| [rq_gi_quadrants_experiments.py](rq_gi_quadrants_experiments.py) | **4-quadrant diagnostic**: Maps (Diversity, Tension) space, topology sweeps, radicalization trajectories, phase diagrams over mixing/intolerance |
| [test_gdrq_1.py](test_gdrq_1.py) | Integration test for `gdrq_1` package |
| [test_csbm.py](test_csbm.py) | CSBM generator validation |
| [benchmark_variance.py](benchmark_variance.py) | Performance comparison of variance computation methods |

---

## Quick Start

### Installation

Requires Python 3.8+ with:
```powershell
pip install numpy matplotlib networkx pandas
```

### Basic Usage

```python
import gdrq_1 as gg
import numpy as np

# Generate a 2-community graph with contextual signal
data = gg.generate_csbm(n=100, p_in=0.3, p_out=0.05, mu=2.0, sigma=0.5)
W = data.W          # Adjacency matrix
x = data.features   # Node signal (correlated with community)

# Compute dispersion metrics
tv = gg.weighted_total_variation(x, W)        # Graph total variation (L1)
de = gg.dirichlet_quadratic(x, W)             # Dirichlet energy (L2)
rq = gg.rayleigh_quotient(x, W)               # Roughness per signal scale

print(f"TV={tv:.3f}, DE={de:.3f}, RQ={rq:.3f}")

# Check Cauchy-Schwarz bound: TV^2 <= (weighted_mass) * DE
bound = gg.cauchy_schwarz_bound_weighted(x, W)
print(f"CS bound holds: {bound['holds']}")
```

### Running Experiments

```powershell
# Echo chambers vs. integration diagnostics
python use_case_sn.py

# 4-quadrant social dynamics analysis
python rq_gi_quadrants_experiments.py

# CSBM spectral recovery
python test_gdrq_1.py
```

---

## Core Concepts

### 1. Complete Graph Identities

On the complete graph $K_n$ (all pairs connected), exact relationships emerge:

$$
\mathcal{E}_D^{K_n}(x) = n^2 \, \text{Var}(x)
$$

$$
G(x) = \frac{1}{2n^2\mu} \sum_{i,j} |x_i - x_j|
$$

For centered signals, the Rayleigh quotient degenerates to $\mathcal{R}_{K_n}(x_c) = n$.

### 2. Cauchy-Schwarz Bounds

Link $L_1$ (Gini-type) and $L_2$ (Dirichlet) graph dispersions:

$$
\text{TV}_W(x)^2 \le \left(\sum_{i<j} w_{ij}\right) \cdot \mathcal{E}_D(x)
$$

Implementation: `cauchy_schwarz_bound_weighted(x, W)['holds']` returns `True` if bound satisfied.

### 3. Spectral Dynamics

Under heat diffusion $\dot{x}(t) = -Lx(t)$:

$$
\text{Var}(x(t)) \le e^{-2\lambda_2 t} \, \text{Var}(x(0))
$$

where $\lambda_2$ (spectral gap) controls exponential smoothing. Gini index admits an exponential upper envelope via $L_1/L_2$ conversion.

### 4. Master Map of Social Network States

This framework defines four **archetypal social states** along two dimensions (Diversity and Tension), with RQ as the critical third dimension:

| State | Diversity<br/>(Gini/Var) | Tension<br/>(Dirichlet) | RQ<br/>(Efficiency) | Diagnosis | Outcome |
|-------|---------------------------|------------------------|--------------------|-----------|---------|
| **1. Consensus** | Low | Low | Low/Stable | Homogeneity | Stable but stagnant |
| **2. Conflict** | High | High | Maximum | Radicalization | Toxicity or revolution |
| **3. Fracture** | High | Low | Minimum | Segregation (Echo Chambers) | Cold polarization |
| **4. Mediation** | Medium–High | Moderate | Optimal | Pluralism with bridges | Slow evolution + cohesion |

**Key Insights:**
- **Low RQ ≠ Peace**: A low Rayleigh Quotient combined with high Gini signals **structural fracture**, not stability (Fracture vs. Consensus).
- **Dirichlet Energy as cost of connection**: In Conflict, large potential jumps concentrate cost unsustainably. In Mediation, bridge nodes decompose tension into smaller gradients.
- **RQ optimum**: The Mediation state balances diversity and tension efficiently, preventing both total consensus stagnation and revolutionary collapse.

**Diagnostic tool:** `rq_gi_quadrants_experiments.py` operationalizes this map via:
- Quadrant archetypes (4 toy scenarios)
- Topology sweeps (fixed opinions, varying network mixing)
- Transition trajectories (radicalization → fracture → mediation)
- Phase diagrams over $(p_{\text{out}}, \tau_{\text{intolerance}})$
- Dynamics evolution (DeGroot vs. bounded-confidence)

---

## The Mediation Archetype: Bridge Nodes as Stabilizers

**Problem**: How do diverse networks avoid both revolutionary collapse (Conflict) and segregation (Fracture)?

**Solution**: Introduce **intermediate-opinion bridge nodes** connecting opposed factions.

**Mechanism**:
- Bridge nodes hold moderate positions (e.g., $x=0$ between camps at $x=\pm 1$)
- They connect both camps, reducing the density of direct A-B conflict edges
- Dirichlet Energy is **decomposed** into smaller gradients across bridge nodes, lowering peak tension
- The Rayleigh Quotient reaches an **efficiency optimum**: diversity persists, but tension remains manageable

**Result**: The network evolves slowly toward consensus without explosive polarization or cryogenic segregation.

**Experiment**: `use_case_sn.py:scenario_mediation()` compares:
- **Neutral mediators** ($x=0$): Symmetrically balanced
- **Biased mediators** ($x=0.4$ or $-0.4$): Asymmetric influence (shows one-sided drift)

This archetype is operationalized in the Master Map's **4th state** and is central to understanding stable, diverse networks in practice.

---

## Metric Family: $\mathcal{R}_{p,q}^{(W,\Pi)}(x)$

Unified framework generalizing Rayleigh quotients and Gini-DE ratios:

$$
\mathcal{R}_{p,q}^{(W,\Pi)}(x) = \frac{\|x\|_{p,W}}{\|x\|_{q,\Pi}}
$$

**Recovers:**
- $p=2, q=2, W=L, \Pi=I$: Standard Rayleigh quotient
- $p=1, q=2, W=\text{edges}, \Pi=I$: TV-over-norm ratio
- $p=2, q=1, W=L, \Pi=\text{complete}$: DE-over-Gini "roughness per inequality"

**Implementation:** See `metrics.py:rayleigh_quotient()` and `metrics.py:normalized_rayleigh_quotient()` for $L_2/L_2$ cases; $L_1$ variants via `weighted_total_variation()` + `gini_pairwise_numerator()`.

---

## Design Philosophy

### Conciseness Over Flexibility

Functions compute **one thing well**:
- `dirichlet_quadratic(x, W)` returns a scalar, not a configuration object
- Adjacency validation happens once via `validate_adjacency(W)`
- No class hierarchies; compose functions for complex workflows

### Numerical Stability

| Concern | Mitigation |
|---------|-----------|
| Division by zero | All quotients use `eps=1e-15` guards, returning `nan` on failure |
| Asymmetry in adjacency | `validate_adjacency()` enforces $W = W^T$ |
| Self-loops | Removed by default: `np.fill_diagonal(W, 0)` |
| Centering for spectral | `center_signal()` explicitly removes mean before RQ |

### Differentiability

For GNN regularizers:
- Exact Gini: $O(n \log n)$ via sorting (non-differentiable)
- Differentiable proxy: $O(m)$ Monte Carlo pair sampling or $O(|E|)$ local graph-TV

See [gini_and_rq_gpt.md](doc/gini_and_rq_gpt.md) §4 for approximation schemes.

---

## Configuration Reference

### CSBM Parameters

```python
generate_csbm(
    n=200,                # Number of nodes
    p_in=0.15,            # Intra-block edge probability
    p_out=0.03,           # Inter-block edge probability
    mu=1.0,               # Signal strength (Gaussian mean = mu * label)
    sigma=1.0,            # Signal noise (Gaussian std)
    balanced=True,        # Force 50/50 block split
    seed=None             # RNG seed for reproducibility
)
```

**Returns:** `CSBMData(W, labels, features, blocks)`

### Fiedler Vector Extraction

```python
lambda2, v2 = fiedler_vector(W, normalized=False)
# normalized=True  → Symmetric normalized Laplacian
# normalized=False → Combinatorial Laplacian
```

---

## Contributing

This is a **research artifact** (PostDoc project). For production use:
1. Vectorize pairwise operations where possible (current bottleneck)
2. Add sparse matrix support (currently dense NumPy arrays)
3. Implement approximate nearest-neighbor Gini (for graphs with $n > 10^4$)

See `benchmark_variance.py` for performance profiling template.

---

## References

### Theoretical Foundations

- **Gini Index:** Lorenz curves, mean absolute difference, $L_1$-dispersion theory
- **Dirichlet Energy:** Laplacian quadratic form, heat equation, spectral graph theory
- **Rayleigh Quotient:** Variational characterization of eigenvalues, spectral clustering
- **Contextual SBM:** Community detection with node features, semi-supervised learning

### Key Equations from Documentation

| Identity | Source | Formula |
|----------|--------|---------|
| Complete graph Dirichlet | [gini_and_rq_gpt.md](doc/gini_and_rq_gpt.md) | $\mathcal{E}_D^{K_n}(x) = n^2 \, \text{Var}(x)$ |
| Gini pairwise form | [gini_and_rq_1.md](doc/gini_and_rq_1.md) | $G(x) = \frac{1}{2n^2\mu} \sum_{i,j} \|x_i - x_j\|$ |
| Variance spectral decay | [variance_spectral.md](doc/variance_spectral.md) | $\text{Var}(x(t)) \le e^{-2\lambda_2 t} \, \text{Var}(x(0))$ |
| Frobenius-Gram identity | [gram_and_rq.md](doc/gram_and_rq.md) | $\|X\|_F^2 = \text{Tr}(X^T X)$ |

---

## License

Research code; check with the repository owner before redistribution.
