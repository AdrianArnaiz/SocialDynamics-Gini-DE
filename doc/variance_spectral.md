# 1️⃣ Verification of the Fundamental Identity

You state:

$$
\sum_{i,j} (x_i - x_j)^2 \;=\; 2n\, x^T x \;-\; 2S^2
$$

This is **correct**.

### Proof (compact derivation)

Start from
$$
\sum_{i,j}(x_i - x_j)^2
\;=\;
\sum_{i,j}\bigl(x_i^2 + x_j^2 - 2x_i x_j\bigr).
$$

Split terms:
- $\sum_{i,j} x_i^2 = n \sum_i x_i^2$
- $\sum_{i,j} x_j^2 = n \sum_j x_j^2$
- $\sum_{i,j} x_i x_j = \left(\sum_i x_i\right)\left(\sum_j x_j\right) = S^2$

Thus,
$$
\sum_{i,j}(x_i - x_j)^2
=
n \sum_i x_i^2
+
n \sum_j x_j^2
-
2S^2
=
2n\, x^T x - 2S^2.
$$

✔ Fully correct.

---

# 2️⃣ Relationship to Variance

You give:

$$
\sum_i (x_i - \bar{x})^2
\;=\;
x^T x \;-\; \frac{S^2}{n}.
$$

Correct.

Because
$$
\bar{x} \;=\; \frac{S}{n},
$$
and expanding:
$$
\sum_i (x_i - \bar{x})^2
=
\sum_i x_i^2 - 2\bar{x}\sum_i x_i + \sum_i \bar{x}^2
=
x^T x - 2\bar{x}S + n\bar{x}^2
=
x^T x - \frac{S^2}{n}.
$$

✔ Correct.

---

### Pairwise Form of Variance

You state:

$$
s^2
=
\frac{1}{2n(n-1)}
\sum_{i,j}(x_i - x_j)^2.
$$

Let’s verify consistency.

From the identity:
$$
\sum_{i,j}(x_i - x_j)^2
=
2n\, x^T x - 2S^2
=
2n\left(x^T x - \frac{S^2}{n}\right).
$$

But for the (unbiased) sample variance $s^2$,
$$
s^2
=
\frac{1}{n-1}\sum_i (x_i-\bar{x})^2
=
\frac{1}{n-1}\left(x^T x - \frac{S^2}{n}\right),
$$
so equivalently
$$
x^T x - \frac{S^2}{n} = (n-1)s^2.
$$

Thus,
$$
\sum_{i,j}(x_i - x_j)^2
=
2n(n-1)s^2,
$$
and rearranging gives
$$
s^2
=
\frac{1}{2n(n-1)}
\sum_{i,j}(x_i - x_j)^2.
$$

✔ Exactly correct.

---

# 3️⃣ Matrix Form (More Structural View)

This is where things become powerful.

Define:
$$
\mathbf{1} = (1,\dots,1)^T.
$$

Define the centering matrix:
$$
H = I - \frac{1}{n}\mathbf{1}\mathbf{1}^T.
$$

Then:
$$
\sum_i (x_i - \bar{x})^2
=
x^T H x.
$$

And since for the complete graph $K_n$,
$$
L_{K_n} = nI - \mathbf{1}\mathbf{1}^T,
\qquad
\text{so}\qquad
H = \frac{1}{n}L_{K_n},
$$
we get:
$$
x^T L_{K_n} x
=
n\, x^T H x
=
n\sum_i (x_i-\bar{x})^2.
$$

Also, using the standard Laplacian quadratic form identity
$$
x^T L x = \frac{1}{2}\sum_{i,j} w_{ij}(x_i-x_j)^2,
$$
and for $K_n$ with $w_{ij}=1$ for $i\neq j$, we obtain:
$$
\sum_{i,j}(x_i - x_j)^2
=
2\, x^T L_{K_n} x.
$$

This is **extremely important** for your research.

---

# 4️⃣ Spectral Interpretation

For the complete graph:
$$
L_{K_n} = nI - \mathbf{1}\mathbf{1}^T.
$$

Eigenvalues:
- $0$ (multiplicity $1$, eigenvector $\mathbf{1}$)
- $n$ (multiplicity $n-1$)

So:
$$
x^T L_{K_n} x
=
x^T (nH) x
=
n\, x^T H x.
$$

Since $H$ is an orthogonal projector (symmetric, idempotent), $x^T H x = \|Hx\|_2^2$. Hence
$$
x^T L_{K_n} x
=
n\,\|Hx\|_2^2.
$$

Meaning:

### The pairwise dispersion is exactly proportional to the squared norm of the centered signal.

This confirms your geometric statement:
- $(x^T x)$: total energy relative to origin
- $(x^T H x)$: energy orthogonal to the constant direction

This is a **projection decomposition**:
$$
x
=
\underbrace{\bar{x}\mathbf{1}}_{\text{mean component}}
+
\underbrace{Hx}_{\text{dispersion component}}.
$$

Orthogonal decomposition.

---

# 5️⃣ Refinement of Geometric Interpretation

Your interpretation is correct, but can be sharpened.

| Quantity       | True geometric meaning                                  |
| -------------- | -------------------------------------------------------- |
| $(x^T x)$      | squared Euclidean norm in $\mathbb{R}^n$                 |
| $(S^2)$        | squared projection onto $\mathrm{span}\{\mathbf{1}\}$     |
| $(x^T H x)$    | squared norm in subspace orthogonal to $\mathbf{1}$       |
| Pairwise sum   | Dirichlet energy on $K_n$                                |

Thus:
$$
x^T x
=
\underbrace{\|Hx\|_2^2}_{\text{dispersion}}
+
\underbrace{\frac{S^2}{n}}_{\text{mean energy}}.
$$

This is a **Pythagorean decomposition**.

---

# 6️⃣ Computational Remarks (Minor Refinement)

You mention possible instability.

More precisely:

The instability arises from computing
$$
x^T x - \frac{S^2}{n}
$$
when both terms are large and nearly equal.

This is catastrophic cancellation.

Stable algorithms use:
- Two-pass algorithm
- Welford’s online algorithm

NumPy internally uses stable implementations.

---

# 7️⃣ Deeper Conceptual Insight (Important)

What you derived is:

$$
\textbf{Variance}
\;=\;
\frac{1}{n-1}\,\|Hx\|_2^2.
$$

$$
\textbf{Pairwise dispersion}
\;=\;
2\, x^T L_{K_n} x.
$$

This means:

> Variance is Dirichlet energy on the complete graph, scaled.

This is precisely the bridge toward:
- Rayleigh quotient
- Graph Laplacians
- Gini index (L1 version)
- Polarization metrics

---

# 8️⃣ Connection to Your Broader Research

Given your work on:
- Dirichlet energy
- Rayleigh quotients
- Polarization
- Social tension metrics

We can reinterpret:
- Variance = quadratic polarization on $K_n$
- Graph Dirichlet energy = structured polarization
- Gini index = L1 polarization

So your document is essentially:

> The L2 polarization theory on the complete graph.

Which is the quadratic analogue of Gini.

