import numpy as np
import time

def calculate_variances(x):
    n = len(x)
    
    # Metodo 1: NumPy estándar (Referencia)
    start = time.time()
    var_numpy = np.var(x, ddof=1)
    t1 = time.time() - start

    # Metodo 2: Suma de Diferencias al Cuadrado (Pairwise)
    # s2 = sum(sum((xi-xj)^2)) / (2*n*(n-1))
    start = time.time()
    diff_matrix = x[:, np.newaxis] - x
    sum_sq_diff = np.sum(diff_matrix**2)
    var_pairwise = sum_sq_diff / (2 * n * (n - 1))
    t2 = time.time() - start

    # Metodo 3: Producto Punto (xTx) y Suma
    # s2 = (xTx - (sum(x)^2 / n)) / (n-1)
    start = time.time()
    xt_x = np.dot(x, x)
    sum_x = np.sum(x)
    var_xtx = (xt_x - (sum_x**2 / n)) / (n - 1)
    t3 = time.time() - start

    return (var_numpy, var_pairwise, var_xtx), (t1, t2, t3)

# Configuración del experimento
data = np.random.randn(20000)  # 2000 elementos
results, times = calculate_variances(data)

print(f"{'Metodo':<25} | {'Resultado':<15} | {'Tiempo (s)':<10}")
print("-" * 55)
print(f"{'NumPy np.var':<25} | {results[0]:<15.6f} | {times[0]:.6f}")
print(f"{'Pairwise (Diff Squares)':<25} | {results[1]:<15.6f} | {times[1]:.6f}")
print(f"{'Producto Punto (xTx)':<25} | {results[2]:<15.6f} | {times[2]:.6f}")
