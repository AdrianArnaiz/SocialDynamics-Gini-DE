# Gini Index vs. Dirichlet Energy on Graphs  
*A formal correspondence, spectral dynamics, and a unifying metric family*

## Four most important points (takeaways)

1. **Complete-graph equivalence (global dispersion):** on the complete graph $K_n$, Dirichlet energy is *exactly* scaled variance:  
   $$\mathcal E_D^{K_n}(x)=x^\top L_{K_n}x = n^2\operatorname{Var}(x).$$

2. **Gini is a complete-graph $L_1$ “energy”:** the Gini numerator is an all-pairs (complete-graph) absolute-difference energy:  
   $$G(x)=\frac{1}{2n^2\mu}\sum_{i,j}|x_i-x_j|,$$
   i.e., *mean-normalized global $L_1$ disparity*.

3. **Spectral gap controls inequality collapse under diffusion:** for heat diffusion $\dot x=-Lx$, variance contracts at rate $\lambda_2$, and Gini admits an exponential *upper envelope* at rate $\lambda_2$:  
   $$\operatorname{Var}(x(t))\le e^{-2\lambda_2 t}\operatorname{Var}(x(0)),\qquad G(x(t))\le e^{-\lambda_2 t}\cdot \frac{\sigma(0)}{\sqrt2\,\mu}.$$

4. **New diagnostic family (local roughness per global inequality):** ratios of **edge-based $L_p$ tension** to **pairwise $L_q$ inequality** unify Rayleigh-quotient thinking with Gini-like thinking, and separate “edge-exposed friction” from “global disparity”.

---

## Introduction

A scalar node signal $x\in\mathbb R^n$ on a graph $\mathcal G=(V,E)$ can be studied through two complementary lenses:

- **Global inequality / dispersion**: do node values differ across the population? The **Gini index** $G(x)$ measures mean absolute differences across **all pairs** (an $L_1$ object), normalized by the mean.
- **Structural tension / smoothness**: do node values differ across social ties / interaction edges? The **Dirichlet energy** $\mathcal E_D(x)=x^\top Lx$ measures squared differences across **edges** (an $L_2^2$ object).

These are *not* the same quantity: they weight pairs differently (all pairs vs. edges) and use different norms ($L_1$ vs. $L_2^2$).  
However, they become tightly connected in special regimes (notably $K_n$) and through general inequalities and spectral arguments.

---

# 1. Preliminaries

## 1.1 Graph notation

Let $\mathcal{G}=(V,E)$ be an undirected weighted graph with $|V|=n$.

- Weighted adjacency: $W=(w_{ij})$ with $w_{ij}=w_{ji}\ge 0$  
- Degrees: $d_i=\sum_j w_{ij}$, and $D=\mathrm{diag}(d_1,\dots,d_n)$  
- (Combinatorial) Laplacian: $L=D-W$

For a signal $x\in\mathbb R^n$, define mean and centering:
$$
\mu = \frac{1}{n}\sum_{i=1}^n x_i,\qquad \mathbf 1=(1,\dots,1)^\top,\qquad x_c = x-\mu\mathbf 1.
$$
Variance (population):
$$
\sigma^2 = \mathrm{Var}(x)=\frac{1}{n}\sum_{i=1}^n (x_i-\mu)^2=\frac{1}{n}\|x_c\|_2^2.
$$
Coefficient of variation:
$$
\mathrm{CV}(x)=\frac{\sigma}{\mu}\quad\text{(assume $\mu>0$)}.
$$

---

## 1.2 Dirichlet energy

**Definition 1 (Dirichlet energy).**
$$
\mathcal E_D(x):=x^\top Lx.
$$

**Proposition 1 (edge-sum form).** For symmetric $W$,
$$
x^\top Lx = \frac12\sum_{i,j=1}^n w_{ij}(x_i-x_j)^2.
$$

**Proof.** Expand $x^\top(D-W)x=\sum_i d_ix_i^2-\sum_{i,j}w_{ij}x_ix_j$ and use $d_i=\sum_jw_{ij}$ to symmetrize into $\frac12\sum_{i,j}w_{ij}(x_i-x_j)^2$. $\square$

---

## 1.3 Rayleigh quotient (RQ)

**Definition 2 (Laplacian Rayleigh quotient).**
$$
\mathcal R_L(x):=\frac{x^\top Lx}{x^\top x}\quad (x\neq 0).
$$
For inequality/dispersion, it is natural to use the centered version:
$$
\mathcal R_L(x_c):=\frac{x_c^\top Lx_c}{\|x_c\|_2^2},
$$
since shifts by $\mathbf 1$ do not represent inequality.

---

## 1.4 Gini index (GI)

Assume $x_i\ge 0$ and $\mu>0$ (standard inequality setting).

**Definition 3 (Gini index: pairwise form).**
$$
G(x):=\frac{1}{2n^2\mu}\sum_{i=1}^n\sum_{j=1}^n |x_i-x_j|.
$$

**Proposition 2 (exact $O(n\log n)$ formula via sorting).**  
If $x_{(1)}\le\cdots\le x_{(n)}$ is the sorted sequence, then
$$
G(x)=\frac{\sum_{i=1}^n (2i-n-1)\,x_{(i)}}{n\sum_{i=1}^n x_i}.
$$
This is exact and computable in $O(n\log n)$ time.

---

# 2. The complete graph $K_n$: exact correspondence between $L_1$ and $L_2^2$ dispersions

## 2.1 Laplacian of $K_n$

**Proposition 3.** For the unweighted complete graph $K_n$,
$$
L_{K_n}=nI-\mathbf 1\mathbf 1^\top.
$$

**Proof.** In $K_n$, $A=J-I$ with $J=\mathbf 1\mathbf 1^\top$ and degrees $n-1$, so $L=(n-1)I-(J-I)=nI-J$. $\square$

---

## 2.2 Dirichlet energy equals scaled variance on $K_n$

**Theorem 1 (DE–variance identity).**
$$
x^\top L_{K_n}x = n\|x_c\|_2^2 = n^2\mathrm{Var}(x).
$$

**Proof.**
$$
x^\top L_{K_n}x = x^\top(nI-\mathbf 1\mathbf 1^\top)x = n\|x\|_2^2-(\mathbf 1^\top x)^2
= n\sum_i x_i^2 - n^2\mu^2
= n\sum_i (x_i-\mu)^2.
$$
$\square$

**Interpretation.** On $K_n$, Dirichlet energy is *global dispersion* (variance), not just local edge roughness.

---

## 2.3 Gini numerator as complete-graph $L_1$ “energy”

Define the complete-graph $L_1$ pairwise energy:
$$
\mathcal E^{K_n}_{L_1}(x):=\frac12\sum_{i,j}|x_i-x_j|.
$$
Then by Definition 3,
$$
G(x)=\frac{\mathcal E^{K_n}_{L_1}(x)}{n^2\mu}.
$$

So on $K_n$:
- $\mathcal E^{K_n}_{L_1}(x)$ is the **global $L_1$ dispersion**,
- $\mathcal E_D^{K_n}(x)$ is the **global $L_2^2$ dispersion** (variance).

---

## 2.4 Universal $L_1$–$L_2$ bounds: Gini vs. CV (and hence vs. $\sqrt{\mathrm{Var}}$)

The next inequality is **not specific to $K_n$**; it is purely a vector inequality.

**Lemma 1 (all-pairs squared-difference identity).**
$$
\sum_{i,j}(x_i-x_j)^2 = 2n\|x_c\|_2^2 = 2n^2\mathrm{Var}(x).
$$

**Proof.** Expand:
$$
\sum_{i,j}(x_i-x_j)^2=\sum_{i,j}(x_i^2+x_j^2-2x_ix_j)=2n\sum_i x_i^2-2\Big(\sum_i x_i\Big)^2,
$$
and note $\|x_c\|_2^2=\sum_i x_i^2-\frac{1}{n}(\sum_i x_i)^2$. $\square$

**Theorem 2 (Gini–CV bounds; exact inequalities).**  
For $x_i\ge 0$, $\mu>0$,
$$
\frac{1}{\sqrt{2}n}\,\mathrm{CV}(x)\;\le\;G(x)\;\le\;\frac{1}{\sqrt{2}}\,\mathrm{CV}(x).
$$

**Proof.** Let $\Delta_{ij}=x_i-x_j$ over $n^2$ ordered pairs.

- Upper bound by Cauchy–Schwarz:
$$
\sum_{i,j}|\Delta_{ij}|\le \sqrt{n^2}\sqrt{\sum_{i,j}\Delta_{ij}^2}
= n\sqrt{2n^2\sigma^2}=\sqrt2\,n^2\sigma.
$$
Divide by $2n^2\mu$ to get $G(x)\le \sigma/(\sqrt2\,\mu)$.

- Lower bound by $\|z\|_1\ge \|z\|_2$:
$$
\sum_{i,j}|\Delta_{ij}|\ge \sqrt{\sum_{i,j}\Delta_{ij}^2}=\sqrt{2n^2\sigma^2}=\sqrt2\,n\sigma.
$$
Divide by $2n^2\mu$ to get $G(x)\ge \sigma/(\sqrt2\,n\mu)$.

$\square$

**Remark (tightness).** The upper bound is often informative; the lower bound is correct but typically loose without extra distributional assumptions.

---

## 2.5 Rayleigh quotient degeneracy on $K_n$

**Proposition 4.** If $x_c\neq 0$ and $x_c\perp \mathbf 1$, then
$$
\mathcal R_{K_n}(x_c)=n.
$$

**Proof.** Since $\mathbf 1^\top x_c=0$,
$$
L_{K_n}x_c=(nI-\mathbf 1\mathbf 1^\top)x_c=nx_c,
$$
hence $x_c^\top L_{K_n}x_c=n\|x_c\|_2^2$ and the quotient equals $n$. $\square$

**Interpretation.** On maximally connected graphs, centered RQ carries little signal-shape information; it primarily reflects the geometry.

---

# 3. General graphs: DE is edge-local, GI is global

On a general graph $\mathcal G$,
$$
\mathcal E_D(x)=\frac12\sum_{i,j}w_{ij}(x_i-x_j)^2
$$
measures **edge-exposed squared tension**, while
$$
G(x)=\frac{1}{2n^2\mu}\sum_{i,j}|x_i-x_j|
$$
measures **global absolute disparity** across all pairs.

Thus:
- **High $G$, low $\mathcal E_D$** can occur when inequality aligns with communities (few cross edges).
- **Low/moderate $G$, high $\mathcal E_D$** can occur when values oscillate across many edges (high local friction).

A standard spectral bridge between DE and variance is:

**Proposition 5 (spectral sandwich / Poincaré inequality for centered signals).**  
If the graph is connected, then for $x_c\perp \mathbf 1$,
$$
\lambda_2\|x_c\|_2^2 \le x_c^\top Lx_c \le \lambda_n\|x_c\|_2^2.
$$

So DE is “variance times a geometry factor”:
$$
n\lambda_2\,\mathrm{Var}(x)\le \mathcal E_D(x_c)\le n\lambda_n\,\mathrm{Var}(x).
$$

Combined with Theorem 2, you can relate $\mathcal E_D$ and $G$ *up to spectral factors* (and mean normalization).

---

# 4. Spectral inertia: how $\lambda_2$ controls inequality collapse under diffusion

Consider heat diffusion:
$$
\dot x(t)=-Lx(t),\qquad x(t)=e^{-tL}x(0).
$$
Mean is preserved because $L\mathbf 1=0$:
$$
\mu(t)=\mu(0).
$$
Centering gives $x_c(t)=e^{-tL}x_c(0)$.

**Theorem 3 (variance contraction governed by $\lambda_2$).**
$$
\|x_c(t)\|_2^2 \le e^{-2\lambda_2 t}\|x_c(0)\|_2^2,
\qquad
\mathrm{Var}(x(t))\le e^{-2\lambda_2 t}\mathrm{Var}(x(0)).
$$

**Corollary 1 (Gini upper envelope decays at rate $\lambda_2$).**  
Using Theorem 2 and $\mu(t)=\mu$,
$$
G(x(t))\le \frac{1}{\sqrt2}\frac{\sigma(t)}{\mu}
\le \frac{1}{\sqrt2}\frac{e^{-\lambda_2 t}\sigma(0)}{\mu}.
$$

**Important nuance.** This is an exponential *upper envelope* derived from $L_2$ contraction plus an $L_1$–$L_2$ inequality; it does not claim tightness for all signals.

**Proposition 6 (Dirichlet energy as instantaneous variance dissipation).**
$$
\frac{d}{dt}\|x_c(t)\|_2^2 = -2\,x_c(t)^\top Lx_c(t) = -2\mathcal E_D(x_c(t)).
$$

So $\mathcal E_D$ is the instantaneous “pressure” driving dispersion down; $\lambda_2$ controls the slowest mode of that pressure.

---

# 5. Efficient and differentiable Gini approximations for GNN regularization

## 5.1 Exact $O(n\log n)$ Gini (sorting)
Use Proposition 2. This is exact and often sufficient as a loss if you tolerate nondifferentiability at ties (subgradients work in practice).

## 5.2 Differentiable approximations

### (A) Monte Carlo all-pairs differentiable Gini: $O(m)$
Since
$$
G(x)=\frac{\mathbb E_{I,J}[|x_I-x_J|]}{2\mu},\quad I,J\sim\text{Unif}(\{1,\dots,n\}),
$$
estimate with $m$ sampled pairs and a smooth absolute value $\phi_\delta$:
$$
\widehat G_m(x)=\frac{1}{2\mu+\varepsilon}\cdot \frac1m\sum_{k=1}^m \phi_\delta(x_{i_k}-x_{j_k}),
$$
with e.g.
$$
\phi_\delta(z)=\sqrt{z^2+\delta^2}.
$$

### (B) Edge-based “graph-Gini”: $O(|E|)$ (interaction-weighted inequality)
$$
G_E(x)=\frac{1}{2\mu+\varepsilon}\cdot\frac{1}{|E|}\sum_{(i,j)\in E}\phi_\delta(x_i-x_j).
$$
This is not the classical global Gini, but it is often the right geometry-aware regularizer for social graphs.

### (C) DE-based proxy: $O(|E|)$ (stable)
$$
\widetilde G_{DE}(x)=\frac{\sqrt{x_c^\top Lx_c+\varepsilon}}{(2\mu+\varepsilon)\,c},
$$
where $c$ calibrates scale.

---

# 6. “Is RQ equal to DE divided by Gini?”

Not exactly.

- Rayleigh quotient:
$$
\mathcal R_L(x_c)=\frac{x_c^\top Lx_c}{\|x_c\|_2^2},
$$
i.e., **DE divided by quadratic dispersion** (variance/energy), geometry-normalized.

- The ratio
$$
\frac{\mathcal E_D(x)}{G(x)}
=
\frac{x^\top Lx}{G(x)}
=
\frac{2n^2\mu\,x^\top Lx}{\sum_{i,j}|x_i-x_j|}
$$
is **local squared tension per unit global $L_1$ inequality**.

**Interpretation (useful diagnostic).**
- Large $\mathcal E_D/G$: inequality is strongly edge-exposed (high local friction relative to global disparity).
- Small $\mathcal E_D/G$: inequality is mostly “modular/segregated” (global disparity with limited edge friction).

So: RQ is a DE-over-$L_2^2$ inequality; DE/G is a DE-over-$L_1$ inequality. They are complementary, linked via $L_1$–$L_2$ conversion and topology.

---

# 7. Conclusion: a unifying metric family for inequality–geometry diagnostics

We now define a general family that subsumes:
- Dirichlet energy ($p=2$ edge tension),
- Gini numerator ($q=1$ all-pairs disparity),
- DE-over-Gini diagnostics (roughness per inequality),
- and local-vs-global inequality ratios (trustworthy social networks).

## 7.1 Two kernels: interaction vs. comparison

Let $W=(w_{ij})$ be an **interaction kernel** (often adjacency weights).  
Let $\Pi=(\pi_{ij})$ be a **comparison kernel** (uniform, exposure-weighted, diffusion-weighted), with $\pi_{ij}\ge 0$ and $\sum_{i,j}\pi_{ij}=1$.

Define pairwise energies:
$$
\mathcal E_p^{(W)}(x)=\frac12\sum_{i,j} w_{ij}|x_i-x_j|^p,
\qquad
\mathcal D_q^{(\Pi)}(x)=\sum_{i,j}\pi_{ij}|x_i-x_j|^q.
$$

## 7.2 The family

**Definition 4 (two-kernel $(p,q)$ quotient family).**
$$
\boxed{
\mathcal R_{p,q}^{(W,\Pi)}(x)=
\frac{\mathcal E_p^{(W)}(x)}{\mathcal D_q^{(\Pi)}(x)+\varepsilon}
}
$$
Optional mean normalization (for $x\ge 0$):
$$
\boxed{
\mathcal R_{p,q,\mu}^{(W,\Pi)}(x)=
\frac{\mathcal E_p^{(W)}(x)}{\mu^q\big(\mathcal D_q^{(\Pi)}(x)+\varepsilon\big)}
}
$$

### Special cases (mapping)
1. **Dirichlet energy**: $p=2$, interaction $W$ is adjacency weights:
$$
\mathcal E_2^{(W)}(x)=x^\top Lx.
$$

2. **Gini numerator (global $L_1$ disparity)**: $q=1$, uniform $\Pi$:
$$
\mathcal D_1^{(\Pi_{\text{unif}})}(x)=\frac{1}{n^2}\sum_{i,j}|x_i-x_j|.
$$

3. **DE over Gini-like diagnostic**: $(p,q)=(2,1)$ with $W$ adjacency and $\Pi$ uniform:
$$
\mathcal R_{2,1}^{(W,\Pi_{\text{unif}})}(x)
\approx
\frac{\text{edge squared tension}}{\text{global absolute disparity}}.
$$

4. **Local-vs-global $L_1$ disparity ratio**: $(p,q)=(1,1)$:
$$
\mathcal R_{1,1}^{(W,\Pi_{\text{unif}})}(x)
=
\frac{\sum_{i,j} w_{ij}|x_i-x_j|}{\sum_{i,j}|x_i-x_j|}.
$$

5. **How Rayleigh quotient fits**: classical RQ uses a *node-based* quadratic dispersion denominator $\|x_c\|_2^2$ rather than a pairwise $\mathcal D_2^{(\Pi)}$. Conceptually, it is the “$L_2^2$ version” of the same idea: DE divided by quadratic inequality.

---

# 8. Coverage / mapping check (everything previously defined is included)

- Dirichlet energy $\mathcal E_D=x^\top Lx$: Definition 1 + Proposition 1; reused throughout.
- Rayleigh quotient $\mathcal R_L$ and centered $\mathcal R_L(x_c)$: Definition 2; degeneracy on $K_n$ in Proposition 4; contrasted with DE/G in Section 6.
- Gini pairwise definition: Definition 3; exact $O(n\log n)$ form in Proposition 2; approximations in Section 5.
- $K_n$ Laplacian and DE–variance identity: Proposition 3 + Theorem 1.
- Gini numerator as complete-graph $L_1$ energy: Section 2.3.
- $L_1$–$L_2$ bounds: Lemma 1 + Theorem 2 (revised: now clearly stated as universal vector inequalities).
- Spectral gap / diffusion / oversmoothing: Theorem 3 + Corollary 1; DE dissipation Proposition 6.
- New metric family: Section 7.

---

## References (canonical)

- Fan R. K. Chung, *Spectral Graph Theory*, CBMS 92, 1997.  
- Standard facts on Laplacian quadratic forms, Rayleigh quotients, and heat diffusion appear in most spectral graph theory texts and lecture notes (e.g., Chung; Spielman notes).  
- Classical Gini and mean absolute difference identities are standard in inequality measurement literature; the pairwise formula is widely used in econometrics/statistics.