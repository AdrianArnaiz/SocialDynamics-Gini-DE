import numpy as np

# 1. Definición de la matriz X (3 muestras, 2 características)
X = np.array([[1, 2], 
              [3, 4], 
              [5, 6]])

# 2. Cálculo de la Matriz de Gram (G = X.T @ X)
# Nota: Usamos X.T @ X para obtener relaciones entre características (columnas)
G = X.T @ X

print("Matriz de Gram (X^T X):")
print(G)

# 3. Verificación de Norma L2 y Diagonal de G
col_0 = X[:, 0]
norm_l2_sq = np.linalg.norm(col_0)**2
print(f"\nNorma L2 al cuadrado de la columna 0: {norm_l2_sq:.2f}")
print(f"Elemento G[0,0]: {G[0,0]:.2f}")

# 4. Verificación de Norma de Frobenius y Traza de G
norm_frob_sq = np.linalg.norm(X, ord='fro')**2
trace_g = np.trace(G)
print(f"\nNorma de Frobenius de X al cuadrado: {norm_frob_sq:.2f}")
print(f"Traza de la Matriz de Gram: {trace_g:.2f}")

# 5. Cociente de Rayleigh para un vector w
w = np.array([1, 1])
# Numerador: w^T G w (Varianza proyectada)
# Denominador: w^T w (Norma L2 del vector w)
rayleigh_num = w.T @ G @ w
rayleigh_den = w.T @ w
rayleigh_val = rayleigh_num / rayleigh_den

print(f"\nCociente de Rayleigh para w=[1,1]: {rayleigh_val:.2f}")

# 6. Comprobación de que el autovalor máximo es el máximo del Cociente de Rayleigh
eigvals = np.linalg.eigvals(G)
print(f"Autovalor máximo de G: {np.max(eigvals):.2f}")
