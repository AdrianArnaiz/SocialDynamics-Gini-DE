# Gini Index vs. Dirichlet Energy on Graphs: a Formal Correspondence, Spectral Dynamics, and GNN Regularizers

## Introduction

A scalar node signal $x \in \mathbb{R}^n$ on a graph $\mathcal{G}=(V,E)$ can be studied from two complementary angles:

1. **Inequality / dispersion (global, $L_1$-style):** how different are node values *across the population*?  
   The classical **Gini index** $G(x)$ measures the *mean absolute difference* across **all pairs** of nodes, normalized by the mean.

2. **Smoothness / tension (structural, $L_2$-style):** how different are node values *across social ties / interaction edges*?  
   The **Dirichlet energy** (a Laplacian quadratic form) $\mathcal{E}_D(x)=x^\top Lx$ measures the sum of squared differences across **edges**.

These two objects are not the same: one is **global and $L_1$**, the other **local (edge-weighted) and $L_2^2$**.  
However, they become tightly connected in two key regimes:

- On the **complete graph** $K_n$, the Dirichlet energy equals a scaled **variance**, and the Gini numerator is exactly an $L_1$ complete-graph “pairwise energy”. This yields clean $L_1 \leftrightarrow L_2$ inequalities relating $G(x)$ to $\sqrt{\mathcal{E}_D(x)}$ and the coefficient of variation.
- Under **heat diffusion** $\dot{x}(t)=-Lx(t)$ (the canonical oversmoothing dynamics), the **spectral gap** $\lambda_2$ controls the exponential contraction of $L_2$ dispersion, which in turn provides an exponential *upper envelope* for $G(x(t))$.

This note develops the correspondence carefully, proves the main identities and inequalities, and concludes with a **new metric family** that unifies Rayleigh quotients and Gini-like ratios, plus a practical view toward differentiable Gini approximations and GNN regularization.

---

## Summary (what you will get)

- Exact identities:
  - $\mathcal{E}_D^{K_n}(x)=n^2\operatorname{Var}(x)$ (complete graph)
  - $G(x)=\dfrac{1}{2n^2\mu}\sum_{i,j}|x_i-x_j|$ (pairwise form)
  - For centered signals on $K_n$, $\mathcal{R}_{K_n}(x_c)=n$ (Rayleigh quotient degeneracy on $K_n$)

- Inequalities linking Gini and Dirichlet energy on $K_n$:
  - $G(x)\le \dfrac{1}{\sqrt{2}}\dfrac{\sigma}{\mu}$ (useful upper bound)
  - $G(x)\ge \dfrac{1}{\sqrt{2}n}\dfrac{\sigma}{\mu}$ (correct but weak lower bound)

- Spectral “structural inertia”:
  - $\operatorname{Var}(x(t)) \le e^{-2\lambda_2 t}\operatorname{Var}(x(0))$
  - $G(x(t))$ admits an exponential **upper envelope** $\propto e^{-\lambda_2 t}$ via $L_1/L_2$ conversion

- Practical tooling:
  - Exact $O(n\log n)$ Gini via sorting
  - Differentiable $O(m)$ Monte Carlo Gini (pairs sampling) and $O(|E|)$ local graph-Gini
  - $O(|E|)$ DE-based proxies calibrated to approximate inequality

- Conclusion:
  - A unifying **metric family** $\mathcal{R}_{p,q}^{(W,\Pi)}(x)$ that recovers
    - Rayleigh quotients (variance-normalized Dirichlet energy),
    - DE-over-Gini-style “roughness per inequality” diagnostics,
    - local-vs-global inequality ratios (trustworthy social-network diagnostics).

---

# 1. Preliminaries

## 1.1 Graph notation

Let $\mathcal{G}=(V,E)$ be an undirected weighted graph with $|V|=n$.

- Weighted adjacency: $W=(w_{ij})$, with $w_{ij}=w_{ji}\ge 0$
- Degree: $d_i=\sum_j w_{ij}$, $D=\operatorname{diag}(d_1,\dots,d_n)$
- (Combinatorial) Laplacian: $L=D-W$

Let $x\in\mathbb{R}^n$ be a node signal. Define mean and centering:
$$
\mu = \frac{1}{n}\sum_{i=1}^n x_i,\qquad
\mathbf{1}=(1,\dots,1)^\top,\qquad
x_c = x-\mu\mathbf{1}.
$$
Variance (population version):
$$
\sigma^2 = \operatorname{Var}(x)=\frac{1}{n}\sum_{i=1}^n (x_i-\mu)^2 = \frac{1}{n}\|x_c\|_2^2.
$$

---

## 1.2 Dirichlet energy (Laplacian quadratic form)

**Definition 1 (Dirichlet energy).**  
For an undirected weighted graph,
$$
\mathcal{E}_D(x) := x^\top Lx.
$$

**Proposition 1 (edge-sum form).**  
For symmetric $W$,
$$
x^\top Lx = \frac{1}{2}\sum_{i,j=1}^n w_{ij}(x_i-x_j)^2.
$$

**Proof.**  
Expand:
$$
x^\top Lx = x^\top Dx - x^\top Wx
= \sum_i d_i x_i^2 - \sum_{i,j} w_{ij}x_ix_j.
$$
Since $d_i=\sum_j w_{ij}$,
$$
\sum_i d_i x_i^2 = \sum_{i,j} w_{ij}x_i^2.
$$
Thus
$$
x^\top Lx = \sum_{i,j} w_{ij}x_i^2 - \sum_{i,j} w_{ij}x_ix_j
= \frac{1}{2}\sum_{i,j} w_{ij}(x_i^2 + x_j^2 - 2x_ix_j)
= \frac{1}{2}\sum_{i,j} w_{ij}(x_i-x_j)^2.
$$
$\square$

---

## 1.3 Rayleigh quotient (RQ)

**Definition 2 (Rayleigh quotient).**  
For nonzero $x$,
$$
\mathcal{R}_L(x) := \frac{x^\top Lx}{x^\top x}.
$$
For inequality-like analysis, it is often natural to use the centered version:
$$
\mathcal{R}_L(x_c) := \frac{x_c^\top Lx_c}{\|x_c\|_2^2}.
$$
This removes dependence on global shifts by $\mathbf{1}$.

---

## 1.4 Gini index (GI)

Assume $x_i\ge 0$ and $\mu>0$ (the standard inequality setting).

**Definition 3 (Gini index, pairwise form).**
$$
G(x) := \frac{\sum_{i=1}^n\sum_{j=1}^n |x_i-x_j|}{2n^2\mu}.
$$

This is equivalent to the Lorenz-curve definition for finite samples under the usual assumptions.

---

# 2. The complete graph $K_n$: the clean $L_1$ vs. $L_2^2$ bridge

The complete graph is the setting where “local” equals “global” because every pair is an edge.

## 2.1 Laplacian of $K_n$

**Proposition 2 (Laplacian of the complete graph).**  
For the unweighted complete graph $K_n$,
$$
L_{K_n} = nI - \mathbf{1}\mathbf{1}^\top.
$$

**Proof.**  
In $K_n$, $A=J-I$ where $J=\mathbf{1}\mathbf{1}^\top$ and degrees are $n-1$, so
$$
L = D-A = (n-1)I - (J-I) = nI - J.
$$
$\square$

---

## 2.2 Dirichlet energy on $K_n$ equals scaled variance

**Theorem 1 (DE–variance identity on $K_n$).**  
For any $x\in\mathbb{R}^n$,
$$
x^\top L_{K_n}x = n^2\operatorname{Var}(x) = n\|x_c\|_2^2.
$$

**Proof.**  
Using Proposition 2:
$$
x^\top L_{K_n}x = x^\top (nI-\mathbf{1}\mathbf{1}^\top)x
= n\|x\|_2^2 - (\mathbf{1}^\top x)^2.
$$
But $\mathbf{1}^\top x = n\mu$, so
$$
x^\top L_{K_n}x = n\sum_i x_i^2 - n^2\mu^2
= n\sum_i (x_i^2 - 2\mu x_i + \mu^2)
= n\sum_i (x_i-\mu)^2
= n\|x_c\|_2^2.
$$
Since $\|x_c\|_2^2=n\sigma^2$, we get $x^\top L_{K_n}x=n^2\sigma^2$.  
$\square$

**Interpretation.**  
On $K_n$, Dirichlet energy is exactly a global dispersion statistic (variance), not merely “local roughness”.

---

## 2.3 Gini numerator as a complete-graph $L_1$ “energy”

Define the complete-graph $L_1$ pairwise dispersion:

**Definition 4 (complete-graph $L_1$ pairwise energy).**
$$
\mathcal{E}^{K_n}_{L_1}(x) := \frac{1}{2}\sum_{i,j=1}^n |x_i-x_j|.
$$

Then Gini is exactly:
$$
G(x) = \frac{\mathcal{E}^{K_n}_{L_1}(x)}{n^2\mu}.
$$

So on $K_n$:

- $G(x)$ is a **mean-normalized $L_1$ complete-graph pairwise energy**
- $\mathcal{E}_D(x)$ is an **$L_2^2$ complete-graph pairwise energy** (equivalently variance)

This is the core formal $L_1$ vs. $L_2$ mapping.

---

## 2.4 $L_1 \leftrightarrow L_2$ inequalities: bounding Gini by variance (and DE)

Let $\mathrm{CV}(x)=\sigma/\mu$ be the coefficient of variation.

### Theorem 2 (Gini–CV bounds on $K_n$; exact inequalities, one useful and one weak).
For $x_i\ge 0$, $\mu>0$,
$$
\frac{1}{\sqrt{2}n}\,\mathrm{CV}(x) \;\le\; G(x)\;\le\; \frac{1}{\sqrt{2}}\,\mathrm{CV}(x).
$$

**Proof (upper bound).**  
Apply Cauchy–Schwarz to the vector of all pairwise differences $\Delta_{ij}=x_i-x_j$ over $n^2$ ordered pairs:
$$
\sum_{i,j}|\Delta_{ij}| \le \sqrt{n^2}\sqrt{\sum_{i,j}\Delta_{ij}^2}
= n\sqrt{\sum_{i,j}(x_i-x_j)^2}.
$$
On $K_n$, using the ordered-pairs identity
$$
\sum_{i,j}(x_i-x_j)^2 = 2\,x^\top L_{K_n}x = 2n^2\sigma^2,
$$
so
$$
\sum_{i,j}|x_i-x_j|
\le n\sqrt{2n^2\sigma^2}
= \sqrt{2}\,n^2\sigma.
$$
Divide by $2n^2\mu$:
$$
G(x)\le \frac{\sqrt{2}\,n^2\sigma}{2n^2\mu}=\frac{1}{\sqrt{2}}\frac{\sigma}{\mu}.
$$

**Proof (lower bound).**  
Use $\|z\|_1\ge \|z\|_2$ for the same vector $(\Delta_{ij})_{i,j}$:
$$
\sum_{i,j}|x_i-x_j| \ge \sqrt{\sum_{i,j}(x_i-x_j)^2}
= \sqrt{2n^2\sigma^2}
= \sqrt{2}\,n\sigma.
$$
Divide by $2n^2\mu$:
$$
G(x)\ge \frac{\sqrt{2}\,n\sigma}{2n^2\mu} = \frac{1}{\sqrt{2}n}\frac{\sigma}{\mu}.
$$
$\square$

**Remark (why the lower bound is weak).**  
It is correct but generally not tight; stronger lower bounds need additional assumptions (bounded support, tail restrictions, sparsity/shape constraints).

---

### Corollary 1 (Gini bounded by Dirichlet energy on $K_n$).
Using Theorem 1 and the upper bound in Theorem 2,
$$
G(x)\le \frac{1}{\sqrt{2}}\frac{\sigma}{\mu}
= \frac{1}{\sqrt{2}\mu}\frac{\sqrt{x^\top L_{K_n}x}}{n}.
$$

So, on $K_n$,
$$
G(x) \;\lesssim\; \frac{\sqrt{\mathcal{E}_D(x)}}{n\mu},
$$
with the exact constant $1/\sqrt{2}$ for the derived upper bound.

---

## 2.5 Rayleigh quotient degeneracy on $K_n$

**Proposition 3 (RQ on $K_n$ is constant on the centered subspace).**  
If $x_c \ne 0$ and $x_c\perp \mathbf{1}$, then
$$
\mathcal{R}_{K_n}(x_c) = \frac{x_c^\top L_{K_n}x_c}{\|x_c\|_2^2}=n.
$$

**Proof.**  
For $x_c\perp\mathbf{1}$, we have $(\mathbf{1}\mathbf{1}^\top)x_c=\mathbf{1}(\mathbf{1}^\top x_c)=0$, so
$$
L_{K_n}x_c = (nI-\mathbf{1}\mathbf{1}^\top)x_c = nx_c.
$$
Thus
$$
x_c^\top L_{K_n}x_c = n\|x_c\|_2^2,
$$
giving $\mathcal{R}_{K_n}(x_c)=n$.  
$\square$

**Interpretation.**  
On a maximally connected graph, the RQ carries almost no “inequality shape” information once centered; it mostly reflects the graph geometry.

---

# 3. Beyond $K_n$: why DE and GI measure different “inequalities”

On a general graph:

- $\mathcal{E}_D(x)=\frac{1}{2}\sum_{i,j} w_{ij}(x_i-x_j)^2$ measures **edge-exposed squared disagreement** (local tension).
- $G(x)=\frac{1}{2n^2\mu}\sum_{i,j}|x_i-x_j|$ measures **global absolute disparity** (all-pairs inequality).

Thus they can disagree strongly:

- **High Gini, low DE:** two (or more) internally homogeneous communities at different means, with few inter-community edges.
- **Low/medium Gini, high DE:** noisy alternation of values across edges (high local friction) but overall distribution not extremely unequal.

This distinction is precisely what makes the pair $(G, \mathcal{E}_D)$ useful for trustworthy social-network analysis: it separates *population-level inequality* from *tie-level tension*.

---

## 3.1 A geometry-aware “Gini-like” functional (proposed)

To incorporate “who compares to whom”, define pair weights $\Pi=(\pi_{ij})$ with $\pi_{ij}\ge 0$ and $\sum_{i,j}\pi_{ij}=1$.

**Definition 5 (pair-weighted absolute disparity).**
$$
\mathrm{MAD}_\Pi(x) := \sum_{i,j}\pi_{ij}|x_i-x_j|.
$$

A mean-normalized inequality index:
$$
G_\Pi(x) := \frac{\mathrm{MAD}_\Pi(x)}{2\mu}.
$$

Special cases:
- Uniform $\pi_{ij}=1/n^2$ gives standard $G(x)$.
- Edge-weighted $\pi_{ij}\propto w_{ij}$ gives an interaction-weighted inequality (useful in social networks).
- Diffusion-kernel $\pi_{ij}$ (e.g., based on $e^{-tL}$) gives an exposure/reachability-weighted inequality.

---

# 4. Spectral inequality: $\lambda_2$ as “structural inertia” of inequality under diffusion

## 4.1 Heat diffusion / oversmoothing process

Consider the (continuous-time) heat equation on the graph:
$$
\dot{x}(t) = -Lx(t),
\qquad x(0)=x_0.
$$
Solution:
$$
x(t) = e^{-tL}x_0.
$$

Since $L\mathbf{1}=0$ (connected undirected graphs), the mean is preserved:
$$
\mu(t)=\frac{1}{n}\mathbf{1}^\top x(t)=\mu(0).
$$
Define the centered signal $x_c(t)=x(t)-\mu\mathbf{1}$; then
$$
x_c(t)=e^{-tL}x_c(0).
$$

---

## 4.2 Exponential variance contraction controlled by $\lambda_2$

Let $(\lambda_k,u_k)$ be Laplacian eigenpairs:
$$
0=\lambda_1 < \lambda_2 \le \cdots \le \lambda_n,
\qquad u_1\propto \mathbf{1}.
$$
Expand:
$$
x_c(0)=\sum_{k=2}^n a_k u_k
\quad\Rightarrow\quad
x_c(t)=\sum_{k=2}^n a_k e^{-\lambda_k t}u_k.
$$

**Theorem 3 (spectral-gap bound for variance decay).**
$$
\|x_c(t)\|_2^2 \le e^{-2\lambda_2 t}\|x_c(0)\|_2^2
\quad\Rightarrow\quad
\operatorname{Var}(x(t)) \le e^{-2\lambda_2 t}\operatorname{Var}(x(0)).
$$

**Proof.**
$$
\|x_c(t)\|_2^2 = \sum_{k=2}^n a_k^2 e^{-2\lambda_k t}
\le e^{-2\lambda_2 t}\sum_{k=2}^n a_k^2
= e^{-2\lambda_2 t}\|x_c(0)\|_2^2.
$$
Divide by $n$ to get the variance statement.  
$\square$

**Interpretation.**  
$\lambda_2$ (the spectral gap / algebraic connectivity) controls the slowest nontrivial decay mode: small $\lambda_2$ implies slow homogenization and persistent disparities; large $\lambda_2$ implies rapid collapse toward consensus.

---

## 4.3 A rigorous exponential upper envelope for Gini under diffusion

We combine:
1) mean preservation ($\mu(t)=\mu(0)$), and  
2) an $L_1/L_2$ inequality of the form $G(x)\le C\cdot \sigma/\mu$ (e.g., the $K_n$ bound gives $C=1/\sqrt{2}$; more generally $C$ depends on how you relate $L_1$ all-pairs differences to an $L_2$ dispersion).

A clean, always-true statement is the following *envelope* bound:

**Corollary 2 (spectral-gap envelope for Gini via variance).**  
If for your setting you use an inequality of the form
$$
G(x)\le C\frac{\sigma}{\mu},
$$
then under heat diffusion,
$$
G(x(t)) \le C\frac{\sqrt{\operatorname{Var}(x(t))}}{\mu}
\le C\,e^{-\lambda_2 t}\frac{\sqrt{\operatorname{Var}(x(0))}}{\mu}.
$$

**Remark (what this does and does not claim).**  
This gives an exponential *upper envelope* decaying at rate $\lambda_2$. It does not claim that $G(x(t))$ equals that envelope or that the bound is tight; it formalizes how topology constrains the fastest possible persistence of inequality under diffusion.

---

## 4.4 DE as instantaneous dissipation of variance

**Proposition 4 (energy dissipation identity).**
$$
\frac{d}{dt}\|x_c(t)\|_2^2 = -2\,x_c(t)^\top Lx_c(t) = -2\mathcal{E}_D(x_c(t)).
$$

**Proof.**
$$
\frac{d}{dt}\|x_c(t)\|_2^2 = \frac{d}{dt}\langle x_c(t),x_c(t)\rangle
= 2\langle x_c(t),\dot{x}_c(t)\rangle.
$$
Since $\dot{x}_c(t)=-Lx_c(t)$,
$$
\frac{d}{dt}\|x_c(t)\|_2^2 = -2\langle x_c(t),Lx_c(t)\rangle = -2x_c(t)^\top Lx_c(t).
$$
$\square$

**Interpretation.**  
Dirichlet energy is the instantaneous “pressure” driving dispersion (variance) down under smoothing.

---

# 5. Algorithmic efficiency: exact and differentiable Gini computations for GNN regularization

## 5.1 Exact $O(n\log n)$ Gini (sorting formula)

Let $x_{(1)}\le\cdots\le x_{(n)}$ denote sorted values.

**Proposition 5 (exact sort-based Gini).**
$$
G(x)=\frac{\sum_{i=1}^n (2i-n-1)\,x_{(i)}}{n\sum_{i=1}^n x_i}.
$$
This computes exact Gini in $O(n\log n)$ time (sorting dominates).

---

## 5.2 Differentiable approximations

### (A) Monte Carlo all-pairs Gini: $O(m)$, differentiable
Using the identity
$$
G(x)=\frac{\mathbb{E}_{I,J}[|x_I-x_J|]}{2\mu},
\qquad I,J \sim \text{Unif}(\{1,\dots,n\}),
$$
estimate with $m$ sampled pairs:
$$
\widehat{G}_m(x) = \frac{1}{2\mu+\varepsilon}\cdot \frac{1}{m}\sum_{k=1}^m \phi_\delta\!\left(x_{i_k}-x_{j_k}\right),
$$
where $\phi_\delta(z)\approx |z|$ is smooth, e.g.
$$
\phi_\delta(z)=\sqrt{z^2+\delta^2}
\quad\text{or}\quad
\phi_\delta(z)=\delta\log\big(2\cosh(z/\delta)\big).
$$

### (B) Edge-based “graph Gini”: $O(|E|)$, differentiable (geometry-aware)
Define
$$
G_E(x) := \frac{1}{2\mu+\varepsilon}\cdot \frac{1}{|E|}\sum_{(i,j)\in E}\phi_\delta(x_i-x_j).
$$
This is not the classical Gini (global), but is often the right *interaction-weighted inequality regularizer* for social graphs.

### (C) DE-based proxy: $O(|E|)$, stable gradients
Use
$$
\widetilde{G}_{DE}(x) := \frac{\sqrt{x_c^\top Lx_c+\varepsilon}}{(2\mu+\varepsilon)\,c},
$$
where $c$ is a calibration constant (fixed or learned) to match scale with $G$.

---

# 6. “RQ as DE/GI?” — the precise statement

The classical Rayleigh quotient is
$$
\mathcal{R}_L(x_c)=\frac{x_c^\top Lx_c}{\|x_c\|_2^2}.
$$
This is exactly a **Dirichlet energy divided by a quadratic dispersion** (variance/energy), not by Gini.

However, the ratio
$$
\frac{\mathcal{E}_D(x)}{G(x)}
= \frac{x^\top Lx}{G(x)}
= \frac{2n^2\mu\cdot x^\top Lx}{\sum_{i,j}|x_i-x_j|}
$$
is still a highly meaningful diagnostic:

- Large $\mathcal{E}_D/G$: inequality is strongly **edge-exposed** (high local tension per unit global inequality)
- Small $\mathcal{E}_D/G$: inequality is mostly **aligned with modular structure** (global disparity with limited edge friction)

So:
- **RQ** is “DE / $L_2^2$-inequality (variance)”
- **DE/GI** is “DE / $L_1$-inequality (mean absolute disparity)”

They are related but not identical; the bridge is the $L_1 \leftrightarrow L_2$ conversion (Section 2.4) and the local-vs-global pairing distinction (Section 3).

---

# 7. Conclusion: a unifying metric family $\mathcal{R}_{p,q}^{(W,\Pi)}(x)$

We now formalize a metric family that subsumes:
- Dirichlet energies ($p=2$ on edges),
- Gini-like disparities ($q=1$ on all pairs),
- Rayleigh quotients (ratio vs. an $L_2^2$ dispersion),
- and “local roughness per global inequality” ratios.

## 7.1 Pairwise $L_p$ energies with two kernels: $W$ vs. $\Pi$

Let $W=(w_{ij})$ be an **interaction kernel** (often the adjacency weights).  
Let $\Pi=(\pi_{ij})$ be a **comparison kernel** (often uniform over all pairs, or exposure-weighted, or diffusion-weighted), with $\pi_{ij}\ge 0$ and $\sum_{i,j}\pi_{ij}=1$.

Define the generalized pairwise energies:
$$
\mathcal{E}_{p}^{(W)}(x) := \frac{1}{2}\sum_{i,j} w_{ij}|x_i-x_j|^p,
\qquad
\mathcal{D}_{q}^{(\Pi)}(x) := \sum_{i,j}\pi_{ij}|x_i-x_j|^q.
$$

## 7.2 The metric family

**Definition 6 (two-kernel $(p,q)$ quotient family).**
$$
\boxed{
\mathcal{R}_{p,q}^{(W,\Pi)}(x)
:=
\frac{\mathcal{E}_{p}^{(W)}(x)}{\mathcal{D}_{q}^{(\Pi)}(x)+\varepsilon}
}
$$
Optionally mean-normalize (for inequality contexts with $x\ge 0$):
$$
\boxed{
\mathcal{R}_{p,q,\mu}^{(W,\Pi)}(x)
:=
\frac{\mathcal{E}_{p}^{(W)}(x)}{\mu^q\left(\mathcal{D}_{q}^{(\Pi)}(x)+\varepsilon\right)}
}
$$

### Key special cases (mapping to the concepts above)

1) **Dirichlet energy** (up to conventions):  
Take $p=2$, $W=w_{ij}$, then $\mathcal{E}_2^{(W)}(x)=x^\top Lx$.

2) **Gini numerator**:  
Take $q=1$, $\Pi$ uniform $\pi_{ij}=1/n^2$, then
$$
\mathcal{D}_1^{(\Pi)}(x) = \frac{1}{n^2}\sum_{i,j}|x_i-x_j|
\propto \text{(Gini numerator)}.
$$

3) **“DE over Gini” diagnostic** (local roughness per global inequality):  
Take $p=2$ with $W$ as adjacency weights, and $q=1$ with uniform $\Pi$:
$$
\mathcal{R}_{2,1}^{(W,\Pi_{\mathrm{unif}})}(x)
\approx
\frac{\text{edge squared tension}}{\text{global absolute disparity}}.
$$

4) **Local-vs-global $L_1$ inequality ratio** (interaction inequality relative to population inequality):  
Take $p=q=1$:
$$
\mathcal{R}_{1,1}^{(W,\Pi_{\mathrm{unif}})}(x)
=
\frac{\sum_{i,j}w_{ij}|x_i-x_j|}{\sum_{i,j}|x_i-x_j|}.
$$
This quantifies the fraction of global disparity that is directly expressed across interactions.

5) **Rayleigh-quotient-like normalization**:  
If you choose a denominator that is quadratic dispersion (variance),
you recover the classical RQ:
$$
\mathcal{R}_L(x_c)=\frac{x_c^\top Lx_c}{\|x_c\|_2^2}.
$$
In the family above, this corresponds to using a *node-based* $L_2^2$ dispersion denominator rather than a pairwise $\mathcal{D}_2^{(\Pi)}$; conceptually, it is still “DE divided by $L_2^2$ inequality”.

### Why this family is useful in trustworthy social networks
By separating:
- the **interaction kernel** $W$ (who influences whom),
- from the **comparison kernel** $\Pi$ (who is compared to whom / whose inequality matters),

$\mathcal{R}_{p,q}^{(W,\Pi)}$ can encode:
- polarization tension (edge-exposed disagreement),
- exposure-weighted inequality (platform visibility),
- fairness-at-distance (diffusion/commute-based comparisons),
- and the mismatch between local friction and global inequity.

---

# 8. Mapping check: where every previously defined object appears in this markdown

Below is a “coverage map” ensuring all previous constructs are included and where they live:

- **Dirichlet energy $\mathcal{E}_D(x)=x^\top Lx$**  
  Included in Definition 1 and Proposition 1 (Section 1.2), reused in Sections 2, 4, 6, 7.

- **Edge-sum identity $\frac12\sum_{ij}w_{ij}(x_i-x_j)^2$**  
  Proposition 1 (Section 1.2).

- **Rayleigh quotient $\mathcal{R}_L(x)=\frac{x^\top Lx}{x^\top x}$ and centered $\mathcal{R}_L(x_c)$**  
  Definition 2 (Section 1.3), degeneracy on $K_n$ in Proposition 3 (Section 2.5), and clarified vs DE/GI in Section 6.

- **Gini pairwise form $G(x)=\frac{\sum_{ij}|x_i-x_j|}{2n^2\mu}$**  
  Definition 3 (Section 1.4) and used throughout.

- **Complete graph Laplacian $L_{K_n}=nI-\mathbf{1}\mathbf{1}^\top$**  
  Proposition 2 (Section 2.1).

- **DE on $K_n$ equals $n^2\operatorname{Var}(x)$**  
  Theorem 1 (Section 2.2).

- **Gini numerator as complete-graph $L_1$ energy**  
  Definition 4 + identity right after (Section 2.3).

- **$L_1$–$L_2$ bounds linking $G$ and CV / $\sqrt{DE}$**  
  Theorem 2 + Corollary 1 (Section 2.4).

- **Spectral diffusion, role of $\lambda_2$, variance decay**  
  Theorem 3 (Section 4.2) + Corollary 2 (Section 4.3).

- **DE as instantaneous dissipation of variance**  
  Proposition 4 (Section 4.4).

- **Algorithmic efficiency: exact $O(n\log n)$ Gini and differentiable approximations ($O(m)$, $O(|E|)$, DE proxy)**  
  Section 5 (5.1 and 5.2).

- **“DE/GI is not RQ, but is meaningful”**  
  Section 6.

- **New metric family $\mathcal{R}_{p,q}^{(W,\Pi)}(x)$**  
  Section 7 (conclusion), with explicit mapping to DE, GI numerator, DE/GI, and Rayleigh-quotient-like normalization.

---

## References (minimal, canonical)

- Fan R. K. Chung. *Spectral Graph Theory*. CBMS Regional Conference Series in Mathematics, 1997.  
- (Classical inequality metrics) Corrado Gini (1912) foundational work; modern treatments in econometrics/inequality measurement texts.
- Standard linear-algebra inequalities: Cauchy–Schwarz and norm relations $\|z\|_1\ge \|z\|_2$.
