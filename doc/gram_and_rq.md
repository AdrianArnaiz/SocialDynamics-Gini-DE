# Informe Técnico: La Matriz de Gram, Normas y el Cociente de Rayleigh

## 1. Definición de la Matriz de Gram
Para una matriz $X \in \mathbb{R}^{m \times n}$, compuesta por vectores columna $\{v_1, v_2, \dots, v_n\}$, la **matriz de Gram** (o Gramiana) se define como el producto de su transpuesta por sí misma:

$$G = X^T X$$

En términos de sus componentes, cada elemento $G_{ij}$ de la matriz representa el producto escalar entre el vector columna $i$ y el vector columna $j$ de $X$:

$$G_{ij} = v_i^T v_j = \langle v_i, v_j \rangle$$

### Propiedades Clave
*   **Simetría:** $G$ es siempre cuadrada y simétrica ($G^T = G$).
*   **Semidefinida Positiva:** Para cualquier vector $w$, $w^T G w \geq 0$. Esto implica que sus autovalores son no negativos ($\lambda \geq 0$).
*   **Invertibilidad:** $G$ es invertible si y solo si las columnas de $X$ son linealmente independientes.

---

## 2. Relación con la Norma $L^2$
La norma $L^2$ (norma euclidiana) de un vector se define como $\|v\|_2 = \sqrt{v^T v}$. La matriz de Gram es, esencialmente, un organizador de normas y ángulos.

### A. Elementos de la Diagonal
La norma $L^2$ al cuadrado de cada columna de $X$ se encuentra en la diagonal principal de la matriz de Gram:
$$G_{ii} = v_i^T v_i = \|v_i\|_2^2$$
Por lo tanto, la norma de la columna $i$ es $\|v_i\|_2 = \sqrt{G_{ii}}$.

### B. Normas de Proyecciones (Formas Cuadráticas)
Si multiplicamos una matriz de datos $X$ por un vector de pesos $w$, la norma $L^2$ al cuadrado del vector resultante ($y = Xw$) se calcula mediante la matriz de Gram:
$$\|Xw\|_2^2 = (Xw)^T (Xw) = w^T (X^T X) w = w^T G w$$

---

## 3. Relación con la Norma de Frobenius
Cuando el análisis se extiende de vectores a matrices, la medida de magnitud estándar es la **Norma de Frobenius** ($\|X\|_F$). Existe una identidad fundamental que vincula esta norma con la traza ($Tr$) de la matriz de Gram:

$$\|X\|_F^2 = \sum_{i=1}^m \sum_{j=1}^n x_{ij}^2 = Tr(X^T X)$$

Dado que la traza es la suma de los elementos de la diagonal, esto confirma que la norma de Frobenius al cuadrado es la suma de las normas $L^2$ al cuadrado de todas las columnas:
$$\|X\|_F^2 = \sum_{i=1}^n \|v_i\|_2^2$$

---

## 4. El Cociente de Rayleigh: Dualidad de Denominadores
El Cociente de Rayleigh se presenta de distintas formas según el contexto de optimización. La confusión sobre el denominador se resuelve identificando el objeto de estudio:

### Caso 1: Optimización de un Vector (e.g., PCA, Autovalores)
Cuando buscamos maximizar la varianza proyectada sobre un vector $w$:
$$R(G, w) = \frac{w^T G w}{w^T w} = \frac{w^T (X^T X) w}{\|w\|_2^2}$$
*   **Denominador:** Es la **norma $L^2$ al cuadrado** del vector de pesos $\|w\|_2^2$.
*   **Significado:** Se busca la dirección de máxima varianza normalizada por la longitud del vector de dirección.

### Caso 2: Optimización de una Matriz
En problemas de alineación matricial o cuando se busca una matriz $X$ que maximice una forma cuadrática:
$$R(A, X) = \frac{Tr(X^T A X)}{Tr(X^T X)} = \frac{Tr(X^T A X)}{\|X\|_F^2}$$
*   **Denominador:** Es la **norma de Frobenius al cuadrado** $\|X\|_F^2$.
*   **Nota:** Como $\|X\|_F^2 = Tr(X^T X)$, aquí el denominador **es** la traza de la matriz de Gram de $X$.

### Caso 3: Métrica Generalizada
En contextos de Mínimos Cuadrados Generalizados, el denominador puede ser una forma cuadrática respecto a una matriz $B$:
$$R(w) = \frac{w^T A w}{w^T B w}$$
Aquí, $B$ suele ser la **matriz de Gram** de un conjunto de datos diferente, actuando como una métrica de normalización específica.

---

## 5. Tabla Resumen de Identidades


| Concepto | Fórmula | Relación con $X^T X$ |
| :--- | :--- | :--- |
| **Norma $L^2$ (Columna $i$)** | $\|v_i\|_2$ | $\sqrt{G_{ii}}$ |
| **Norma de Frobenius de $X$** | $\|X\|_F$ | $\sqrt{Tr(G)}$ |
| **Distancia Euclidiana** | $d(v_i, v_j)^2$ | $G_{ii} + G_{jj} - 2G_{ij}$ |
| **Ángulo (Coseno)** | $\cos(\theta)$ | $\frac{G_{ij}}{\sqrt{G_{ii} G_{jj}}}$ |
| **Varianza Proyectada** | $\sigma_{proj}^2$ | $w^T G w$ |
