# A Coherent View of Gini-Type Dispersion, Dirichlet Energy, and the Rayleigh Quotient on Graphs

## S0. Conventions and notation (to avoid factor-of-2 confusion)

We work with an undirected weighted graph $G=(V,E,W)$ with $n=|V|$, symmetric weights $w_{ij}=w_{ji}\ge 0$, and $w_{ii}=0$. Let $x\in\mathbb{R}^n$ be a scalar signal on the nodes.

Define:
- Degree $d_i=\sum_j w_{ij}$
- Degree matrix $D=\mathrm{diag}(d_i)$
- Adjacency matrix $W=(w_{ij})$
- (Combinatorial) graph Laplacian $L=D-W$

For a complete unweighted graph $K_n$, we have $w_{ij}=1$ for $i\ne j$.

### Pairwise-sum conventions
For undirected graphs, it is often cleaner to sum over unique pairs $i<j$. When summing over all ordered pairs $(i,j)$, each edge is counted twice. In particular,

$$
\sum_{i,j}|x_i-x_j| = 2\sum_{i<j}|x_i-x_j|,
\qquad
\frac12\sum_{i,j}w_{ij}(x_i-x_j)^2 = \sum_{i<j}w_{ij}(x_i-x_j)^2.
$$

Throughout, we will state formulas with explicit conventions.

---

## S1. The three core objects: $L_1$-type dispersion, $L_2$-type dispersion, and the Rayleigh quotient

### S1.1. Gini-type pairwise dispersion ($L_1$ on pairwise differences)

The (unnormalized) Gini-type numerator on a complete graph can be written as

$$
G_{\mathrm{num}}(x)=\sum_{i<j}|x_i-x_j|.
$$

(If one uses $\sum_{i,j}$, this is exactly $2G_{\mathrm{num}}(x)$.)

This is a **linear** measure of pairwise dispersion. It is robust to outliers relative to quadratic penalties, but it is mathematically less convenient because $|t|$ is not differentiable at $t=0$.

In graph terms, this is an $L_1$-type norm of edge differences (equivalently, graph total variation on the complete graph).

---

### S1.2. Dirichlet energy ($L_2$ on pairwise differences)

For a weighted graph, define the quadratic Dirichlet numerator (quadratic form)

$$
Q_W(x) := x^\top L x
= \frac12\sum_{i,j} w_{ij}(x_i-x_j)^2
= \sum_{i<j} w_{ij}(x_i-x_j)^2.
$$

This is the graph analogue of quadratic dispersion (an $L_2$-type energy on edge differences). Since
$$
(x_i-x_j)^2 = |x_i-x_j|^2,
$$
the key difference with the Gini-type numerator is not the underlying differences themselves, but the **penalty shape**:
- $L_1$-type (Gini): linear in $|x_i-x_j|$
- $L_2$-type (Dirichlet): quadratic in $|x_i-x_j|$, so large differences are penalized much more strongly

---

### S1.3. Rayleigh quotient (RQ) and the denominator $x^\top x$

The (unnormalized Laplacian) Rayleigh quotient is

$$
R_L(x)=\frac{x^\top L x}{x^\top x}, \qquad x\neq 0.
$$

Its denominator is the squared Euclidean norm

$$
x^\top x = \|x\|^2 = \sum_i x_i^2.
$$

Interpretation:
- The numerator measures graph roughness / lack of smoothness.
- The denominator fixes the overall scale of $x$, preventing trivial scaling effects.
- In spectral methods, one also imposes an orthogonality/centering constraint (e.g., $x\perp \mathbf{1}$) to exclude the constant-vector mode.

> **Remark (normalized spectral methods):** In many graph learning / clustering settings, one uses the normalized Laplacian and a denominator like $x^\top D x$. Here we keep the $\,x^\top x$ version because it is the one directly discussed in your notes.

---

## S2. Exact identities on the complete graph $K_n$

Assume $K_n$ is unweighted, so $w_{ij}=1$ for $i\ne j$.

### S2.1. Pairwise-square identity (complete graph)

The Dirichlet quadratic numerator becomes

$$
Q_{K_n}(x)=\frac12\sum_{i,j}(x_i-x_j)^2=\sum_{i<j}(x_i-x_j)^2.
$$

A standard identity gives

$$
Q_{K_n}(x)
= n\sum_i x_i^2-\left(\sum_i x_i\right)^2.
$$

Equivalently, if $\bar x=\frac1n\sum_i x_i$,

$$
Q_{K_n}(x)=n\sum_i (x_i-\bar x)^2.
$$

So on the complete graph, Dirichlet energy is exactly a pairwise form of variance (up to scaling).

---

### S2.2. Centered signals and the RQ denominator

If the signal is centered, i.e.
$$
\sum_i x_i = 0,
$$
then the complete-graph identity simplifies to

$$
Q_{K_n}(x)=n\|x\|^2 = n(x^\top x).
$$

Therefore, for centered $x$,

$$
\frac12\sum_{i,j}(x_i-x_j)^2 = n(x^\top x).
$$

This is the precise constant-factor relationship between:
- the complete-graph Dirichlet numerator, and
- the denominator of the (unnormalized) Rayleigh quotient.

---

### S2.3. What each quantity measures (comparison table)

| Measure | Representative numerator / term | Sensitivity profile | What it emphasizes |
|---|---|---|---|
| **Gini-type (linear)** | $\sum_{i<j}\|x_i-x_j\|$ | **Linear** in pairwise gaps | Typical pairwise disparity |
| **Dirichlet (quadratic)** | $\sum_{i<j}(x_i-x_j)^2$ (or $\frac12\sum_{i,j}(x_i-x_j)^2$) | **Quadratic** (outlier-sensitive) | Large jumps / roughness |
| **RQ denominator** | $\sum_i x_i^2$ | Scale term | Signal magnitude (normalization) |

---

### S2.4. The “quadratic Gini” interpretation of the RQ (and an important caveat)

You can view the Rayleigh quotient as a **quadratic analogue** of a pairwise-dispersion-to-scale ratio:

$$
R_L(x)
=
\frac{x^\top L x}{x^\top x}
=
\frac{\frac12\sum_{i,j}w_{ij}(x_i-x_j)^2}{\sum_i x_i^2}.
$$

On a complete graph, the numerator compares **all pairs** (because all nodes are neighbors). The denominator normalizes the scale.

**Important caveat:** on the unweighted complete graph, if $x\perp \mathbf{1}$ (equivalently centered), then
$$
R_L(x)=n,
$$
so the quotient is constant on that subspace. In other words, the formula is correct and insightful, but on $K_n$ it is not discriminative once you impose the usual centering constraint.

---

## S3. Cauchy–Schwarz bridge: linking the Gini numerator and Dirichlet numerator

The key inequality relating $L_1$-type and $L_2$-type pairwise dispersion is **Cauchy–Schwarz**:

$$
\left(\sum_k u_k v_k\right)^2 \le \left(\sum_k u_k^2\right)\left(\sum_k v_k^2\right).
$$

This gives a clean upper bound of the Gini-type numerator in terms of the Dirichlet numerator.

### S3.1. Complete graph (unweighted): global pairwise inequality

Let $N=\frac{n(n-1)}{2}$ be the number of unique pairs $(i,j)$ with $i<j$. Define $d_k=|x_i-x_j|$ over the $N$ pairs.

Apply Cauchy–Schwarz with
$$
u_k=d_k=|x_i-x_j|,
\qquad
v_k=1.
$$

Then

$$
\left(\sum_{i<j}|x_i-x_j|\right)^2
\le
\left(\sum_{i<j}(x_i-x_j)^2\right)\left(\sum_{k=1}^N 1\right)
=
N\sum_{i<j}(x_i-x_j)^2.
$$

That is,

$$
\left(G_{\mathrm{num}}(x)\right)^2
\le
\frac{n(n-1)}{2}\;Q_{K_n}(x)
$$

or equivalently

$$
G_{\mathrm{num}}(x)
\le
\sqrt{\frac{n(n-1)}{2}\;Q_{K_n}(x)}
.
$$

#### Consequences (complete graph)
1. **Consistency:** If the quadratic dispersion $Q_{K_n}(x)$ is small, then the Gini-type dispersion must also be small.
2. **Sensitivity contrast:** $Q_{K_n}$ scales quadratically with differences, while $G_{\mathrm{num}}$ scales linearly.
3. **Equality condition (restrictive):** Equality in Cauchy–Schwarz requires all pairwise magnitudes $|x_i-x_j|$ (over $i<j$) to be equal on the set of counted pairs, which is highly restrictive.

---

### S3.2. Direct connection to $\|x\|^2$ on a centered complete graph

If $x$ is centered, then $Q_{K_n}(x)=n\|x\|^2$. Substituting into the bound gives

$$
G_{\mathrm{num}}(x)
\le
\sqrt{\frac{n(n-1)}{2}\cdot n\|x\|^2}
=
n\sqrt{\frac{n-1}{2}}\;\|x\|.
$$

Equivalently,

$$
G_{\mathrm{num}}(x)
\le
\sqrt{\frac{n^2(n-1)}{2}\;\|x\|^2}
.
$$

This links the Gini-type numerator directly to the denominator of the Rayleigh quotient (under centering on $K_n$).

---

### S3.3. Weighted arbitrary graphs: topology enters the constant

Now let the graph be arbitrary and weighted. Define the weighted Gini-type numerator (graph total variation)

$$
TV_W(x) := \sum_{i<j} w_{ij}|x_i-x_j|,
$$

and the quadratic Dirichlet numerator

$$
Q_W(x)=\sum_{i<j} w_{ij}(x_i-x_j)^2 = x^\top Lx.
$$

Apply Cauchy–Schwarz to the edge-indexed quantities
$$
\sqrt{w_{ij}}
\quad\text{and}\quad
\sqrt{w_{ij}}\,|x_i-x_j|.
$$

Then

$$
\left(\sum_{i<j} w_{ij}|x_i-x_j|\right)^2
\le
\left(\sum_{i<j} w_{ij}\right)
\left(\sum_{i<j} w_{ij}(x_i-x_j)^2\right).
$$

Hence,

$$
TV_W(x)^2 \le |E|_w \, Q_W(x)
\qquad
\text{where}\quad
|E|_w:=\sum_{i<j}w_{ij}.
$$

If you prefer the convention $\mathcal E_W(x)=\frac12 x^\top Lx$, then the same inequality becomes
$$
TV_W(x)^2 \le 2|E|_w\,\mathcal E_W(x).
$$

#### Key implications (general graph)
1. **From statistical to topological:** The connection is no longer purely “global statistics”; the graph topology determines which pairwise differences matter.
2. **Local vs. global inequality:** On sparse graphs, $TV_W$ measures **local** disparities across edges, not all pairwise disparities.
3. **Community blindness across weak links:** If inter-community edges have tiny weights, both $TV_W$ and $Q_W$ can remain small even when the two communities have very different average values.
4. **Low RQ implies local smoothness:** Since $Q_W(x)$ is the numerator of the Rayleigh quotient, a low RQ forces (after scale normalization) small weighted differences across connected nodes, which also controls $TV_W(x)$ via the inequality above.

> **Summary of the constant:** On $K_n$, the bridge constant is driven by the number of pairs. On a general weighted graph, it is driven by the total edge mass $|E|_w=\sum_{i<j}w_{ij}$.

---

## S4. Optimization viewpoint: gradients, subgradients, and why $L_2$ is computationally easier

This is the practical reason Dirichlet / Rayleigh formulations dominate classical spectral methods, while $L_1$-type graph objectives are robust but harder to optimize.

### S4.1. Dirichlet quadratic energy: smooth gradient and Laplacian linearity

Recall
$$
Q_W(x)=x^\top Lx=\frac12\sum_{i,j}w_{ij}(x_i-x_j)^2.
$$

For symmetric $L$, the gradient is
$$
\nabla Q_W(x)=2Lx.
$$

Equivalently, if one minimizes $\frac12 Q_W(x)$, then
$$
\nabla \!\left(\frac12 Q_W(x)\right)=Lx,
$$
and componentwise
$$
\frac{\partial}{\partial x_i}\left(\frac12 Q_W(x)\right)
=\sum_j w_{ij}(x_i-x_j)
=(Lx)_i.
$$

This linearity is exactly why eigenvalue/eigenvector methods arise naturally.

---

### S4.2. Gini-type / graph-TV objective: nonsmooth subgradient

Define the weighted $L_1$-type graph objective
$$
TV_W(x)=\sum_{i<j} w_{ij}|x_i-x_j|.
$$

This is not differentiable whenever some $x_i=x_j$. One uses a **subgradient**. Away from ties, a componentwise expression is
$$
\frac{\partial TV_W}{\partial x_i}
=
\sum_j w_{ij}\,\mathrm{sgn}(x_i-x_j).
$$

At ties $x_i=x_j$, one replaces $\mathrm{sgn}(0)$ by the subdifferential interval $[-1,1]$.

**Interpretation:** the subgradient depends primarily on relative ordering/sign, not smoothly on the magnitude of the gap. This is one reason $L_1$-type models are:
- more robust to extreme values,
- better at preserving sharp transitions,
- but harder/slower to optimize (e.g., TV-regularization-style problems, proximal/subgradient methods, nonsmooth solvers).

---

## S5. Probability and statistical interpretation (dispersion in distributions)

If the node values $x_i$ are viewed as samples from a random variable $X$, the graph formulas connect to familiar statistical dispersion measures.

### S5.1. Dirichlet / quadratic pairwise dispersion and variance

For i.i.d. copies $X_1,X_2$ of $X$,
$$
\mathbb E[(X_1-X_2)^2]=2\,\mathrm{Var}(X).
$$

So the pairwise quadratic difference is a variance-type quantity (up to a factor of $2$).

For a finite sample on a complete graph, the identity in **S2.1** shows that the complete-graph Dirichlet numerator is exactly proportional to empirical variance.

---

### S5.2. Gini and the mean absolute difference (MAD / Gini mean difference)

The Gini-type pairwise quantity corresponds to the **mean absolute difference** (also called Gini mean difference):

$$
\mathrm{MD}=\mathbb E[|X_1-X_2|].
$$

This often aligns better with an intuitive/human notion of disparity because it grows **linearly** rather than quadratically with pairwise differences.

---

### S5.3. Entropy connection (conceptual, not identity)

There are meaningful conceptual links between dispersion / concentration and entropy, but they are **not equivalences**:

- Maximum entropy under a fixed variance constraint yields the **Gaussian** distribution.
- Gini-type inequality measures and entropy measures (e.g., Shannon, Tsallis) both describe aspects of spread/concentration, but they encode different structures and should not be treated as interchangeable.

---

## S6. Graph cuts, total variation ($L_1$), Dirichlet smoothness ($L_2$), and the Cheeger bridge

This is the frontier between classical graph analysis and geometric learning: the $L_2$ (spectral) world and the $L_1$ / total-variation world are closely related but produce very different behaviors.

### S6.1. Standard Laplacian ($L_2$) and smooth relaxations

The combinatorial Laplacian $L=D-W$ minimizes quadratic roughness:
$$
x^\top Lx=\sum_{i<j} w_{ij}(x_i-x_j)^2.
$$

In spectral clustering, one studies the second eigenvector (or a normalized variant) as a **continuous relaxation** of a discrete partition problem.

- **Effect:** the solution varies smoothly across the graph.
- **Practical issue:** to obtain an actual partition, one still needs a thresholding step (e.g., sign cut, sweep cut).

---

### S6.2. Gini-type $L_1$ objective and graph total variation (TV): sharper cuts

Replacing squares by absolute values gives graph total variation:
$$
TV_W(x)=\sum_{i<j} w_{ij}|x_i-x_j|.
$$

This tends to favor **piecewise-constant** solutions, with sharp jumps at boundaries—often much closer to a true segmentation/cut than an $L_2$-smooth embedding.

For indicator vectors, TV directly recovers cut size:
- If $x=\mathbf{1}_S$ (0/1 indicator), then
  $$
  TV_W(x)=\mathrm{Cut}(S,\bar S).
  $$
- If $x\in\{-1,+1\}^n$ is a signed indicator, then
  $$
  TV_W(x)=2\,\mathrm{Cut}(S,\bar S).
  $$

So the $L_1$-type numerator is directly tied to graph partition boundaries.

---

### S6.3. The Cheeger inequality (precise statement and convention)

Define
$$
\mathrm{Cut}(S,\bar S)=\sum_{i\in S,\;j\notin S} w_{ij},
\qquad
\mathrm{Vol}(S)=\sum_{i\in S} d_i.
$$

A common form of the Cheeger constant (conductance-type normalization) is
$$
h(G)=\min_{S\subset V}\frac{\mathrm{Cut}(S,\bar S)}{\min(\mathrm{Vol}(S),\mathrm{Vol}(\bar S))}.
$$

Let $\mathcal L=I-D^{-1/2}WD^{-1/2}$ be the **normalized Laplacian**, and let $\lambda_2(\mathcal L)$ be its second-smallest eigenvalue. Then (one standard form of) Cheeger’s inequality is

$$
\frac{\lambda_2(\mathcal L)}{2}\le h(G)\le \sqrt{2\lambda_2(\mathcal L)}.
$$

This is the formal bridge between:
- an $L_2$ spectral quantity ($\lambda_2$, via Dirichlet/RQ), and
- a cut quantity closely tied to $L_1$/TV behavior.

> **Computational interpretation:** exact cut optimization (or exact minimum-conductance partitioning) is generally hard, while spectral methods provide a tractable relaxation plus approximation guarantees.

---

### S6.4. Conceptual summary table: diffusion vs segmentation

| Concept | Operator / objective | Local penalty | Typical graph effect |
|---|---|---|---|
| **Dirichlet / spectral** | $x^\top Lx$, Rayleigh quotient | $(x_i-x_j)^2$ | **Diffusion / smoothing** (heat-like spreading) |
| **Gini-type / TV** | $\sum w_{ij}|x_i-x_j|$ | $|x_i-x_j|$ | **Segmentation / sharp boundaries** |
| **RQ denominator** | $x^\top x$ (or $x^\top D x$ in normalized settings) | scale normalization | Prevents trivial scaling and stabilizes optimization |

---

### S6.5. Practical signal-stability interpretation (your research lens)

If you are studying stability or heterogeneity of a graph signal:
- A relatively **high TV$_W$** but only **moderate $Q_W$** suggests many distributed/moderate local differences rather than a few extreme jumps.
- A **high $Q_W$** (especially relative to TV$_W$) indicates that one or a few large discontinuities are dominating, since the quadratic penalty amplifies large jumps.

This is exactly the diagnostic contrast between linear and quadratic dispersion.

---

## S7. Synthesis for algorithm design

If you are designing or analyzing an algorithm on graph signals:

1. **Use a Rayleigh-type denominator** ($x^\top x$, or $x^\top D x$ in normalized settings) to fix the scale.
2. **Use Dirichlet / $L_2$** when you want:
   - smooth solutions,
   - linear gradients,
   - fast spectral/eigenvalue methods,
   - diffusion/regularization behavior.
3. **Use Gini-type / TV / $L_1$** when you want:
   - robustness to outliers,
   - sharper boundaries,
   - piecewise-constant structure,
   - cut/segmentation behavior.

In short: Dirichlet is usually the computationally convenient relaxation; Gini/TV is often closer to the discrete partition/segmentation structure you ultimately care about.

---

## S8. Natural next step (optional derivation)

A natural continuation is to write the derivation in incidence-matrix form:
- $Q_W(x)=\|W^{1/2}Bx\|_2^2$,
- $TV_W(x)=\sum_e w_e |(Bx)_e|$,
and then derive the $L_1$-vs-$L_2$ bounds and Laplacian connections in a single linear-algebra framework.

That gives a clean bridge between:
- pairwise differences,
- graph operators (Laplacian / normalized Laplacian),
- and optimization methods (spectral vs nonsmooth/TV).