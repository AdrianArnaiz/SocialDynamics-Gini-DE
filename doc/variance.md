# Variance Analysis: From Pairwise Differences to Dot Product

This repository documents and verifies the mathematical relationship between:

- The **total energy** of a vector (dot product with itself).
- The **sum of squared pairwise differences**.
- The **sample variance**.
- Their formulation in matrix notation and computationally efficient form.

The central idea is to clearly distinguish between:

- Absolute magnitude (measured from the origin).
- Internal dispersion (measured relative to the data itself).

---

# 1. The Fundamental Identity

Let a dataset be:

$$
x = (x_1, x_2, ..., x_n)
$$

Define: *Dot product* as $x^T x = \sum_{i=1}^{n} x_i^2$, *Sum of elements* as $S = \sum_{i=1}^{n} x_i$ and *mean* as $\bar{x} = \frac{S}{n}$.


---

## 1.1 Sum of Pairwise Differences

The total sum of squared differences between all pairs is:

$$
\sum_{i=1}^{n} \sum_{j=1}^{n} (x_i - x_j)^2
$$

Expanding algebraically:

$$
\sum_{i,j} (x_i - x_j)^2 = 2n \sum_{i=1}^{n} x_i^2 - 2\left(\sum_{i=1}^{n} x_i\right)^2
$$

That is:

$$
\sum_{i,j} (x_i - x_j)^2
=
2n \, x^T x - 2S^2
$$

This is the central algebraic identity of the project.

---

# 2. Relationship to Variance

The sample variance measures dispersion relative to the mean.

## 2.1 Classical Form (Relative to the Mean)

$$
s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2
$$

---

## 2.2 Computational Form (Dot Product)

Using algebra:

$$
\sum_{i=1}^{n} (x_i - \bar{x})^2
=
x^T x - \frac{S^2}{n}
$$

Therefore:

$$
s^2
=
\frac{1}{n-1}
\left(
x^T x - \frac{S^2}{n}
\right)
$$

This formulation avoids constructing large difference matrices and is computationally efficient.

---

## 2.3 Pairwise Form (All-against-all)

Variance can also be expressed using the pairwise sum:

$$
s^2
=
\frac{1}{2n(n-1)}
\sum_{i,j} (x_i - x_j)^2
$$

This form does not require explicitly computing the mean.

---

# 3. Geometric Interpretation

| Concept | Formula | Interpretation |
|----------|----------|----------------|
| Total energy | $x^T x$ | Distance from the origin (absolute magnitude) |
| Sum of differences | $\sum_{i,j} (x_i - x_j)^2$ | Total accumulated separation between all points |
| Variance | $s^2$ | Average internal dispersion |

---

# 4. The Bridge: Centered Data

If the data are centered:

$$
\bar{x} = 0
\quad \Rightarrow \quad S = 0
$$

Then:

$$
x^T x = \sum_{i=1}^{n} x_i^2
$$

And the identity simplifies to:

$$
\sum_{i,j} (x_i - x_j)^2 = 2n \, x^T x
$$

Additionally:

$$
s^2 = \frac{x^T x}{n-1}
$$

When data are centered, total energy directly equals dispersion (up to scaling).

---

# 5. Computational Comparison

| Method | Complexity | Advantages | Disadvantages |
|---------|------------|------------|----------------|
| Pairwise | $O(n^2)$ | Intuitive, directly measures relative distances | Extremely slow and memory-heavy |
| Dot Product | $O(n)$ | Fast and memory-efficient | Can suffer numerical instability with very large values |
| NumPy (`np.var`) | Optimized in C | Industry standard | Conceptually opaque (black box) |

---

# 6. Experimental Results

When running the script:

### Precision
All methods produce the same result (within floating-point precision error).

### Performance
The dot-product method is orders of magnitude faster than the pairwise method.

Reason:

- Pairwise implicitly builds an $n \times n$ structure.
- Dot product operates in linear space $O(n)$.

---

# 7. Conceptual Conclusion

This project demonstrates that:

- Variance is an internal measure of dispersion.
- The pairwise sum is a total measure of separation.
- Both are algebraically connected through the dot product.
- Centering removes global terms and reveals the essential geometric structure.

In summary:

> Absolute magnitude (energy) vs Internal dispersion (variance) are connected through an elegant and computationally efficient algebraic identity.
> 
---