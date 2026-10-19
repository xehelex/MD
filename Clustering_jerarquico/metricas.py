"""Métricas internas y externas para evaluar clustering jerárquico."""
from __future__ import annotations
import numpy as np
from scipy import sparse


def silhouette_medio(D: np.ndarray, etiquetas: np.ndarray) -> float:
    """Silhouette medio a partir de una matriz de distancias simétrica (n, n).

    Devuelve NaN con menos de dos clusters o con un cluster por instancia.
    Las instancias solitarias puntúan 0.
    """
    D = np.asarray(D, dtype=float)
    labels = np.asarray(etiquetas)
    n = len(labels)

    if D.shape != (n, n):
        raise ValueError("D tiene que ser de tamaño (n, n) y encajar con las etiquetas.")

    grupos, inv = np.unique(labels, return_inverse=True)
    k = len(grupos)
    if k < 2 or k >= n:
        return float("nan")

    idx = np.arange(n)
    tam = np.bincount(inv, minlength=k)
    pertenencia = sparse.csr_matrix((np.ones(n), (inv, idx)), shape=(k, n))

    suma = np.asarray(pertenencia @ D).T
    propio = tam[inv]

    a = (suma[idx, inv] - D[idx, idx]) / np.maximum(propio - 1, 1)
    medias = suma / tam
    medias[idx, inv] = np.inf
    b = medias.min(axis=1)

    denom = np.maximum(a, b)
    valores = np.divide(b - a, denom, out=np.zeros(n), where=denom > 0)
    valores[propio == 1] = 0.0

    return float(valores.mean())


def correlacion_cofenetica(resultado) -> float:
    """Correlación de Pearson entre las distancias originales y las alturas del árbol."""
    D = resultado.distancias_originales
    n = resultado.n_instancias
    cophenetica = np.zeros((n, n), dtype=float)

    miembros = {i: [i] for i in range(n)}
    for idx, (izquierda, derecha, altura) in enumerate(resultado.merges):
        i, j = int(izquierda), int(derecha)
        mi, mj = miembros.pop(i), miembros.pop(j)
        for a in mi:
            for b in mj:
                cophenetica[a, b] = cophenetica[b, a] = altura
        miembros[n + idx] = mi + mj

    tri = np.triu_indices(n, 1)
    x, y = D[tri], cophenetica[tri]

    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")

    return float(np.corrcoef(x, y)[0, 1])


def normalizar_filas(X):
    """Escala cada fila a norma 1 (las filas nulas se quedan a cero).

    Sobre vectores normalizados la distancia euclídea es monótona respecto a la coseno.
    """
    if sparse.issparse(X):
        X = X.tocsr().astype(np.float64)
        normas = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
        inversa = np.divide(1.0, normas, out=np.zeros_like(normas), where=normas > 0)
        return sparse.diags(inversa) @ X
    A = np.asarray(X, dtype=float)
    normas = np.linalg.norm(A, axis=1, keepdims=True)
    return np.divide(A, normas, out=np.zeros_like(A), where=normas > 0)


def davies_bouldin(X, etiquetas: np.ndarray) -> float:
    """Índice de Davies-Bouldin euclídeo (menor es mejor).

    Trabaja con centroides y productos escalares, así que no densifica X.
    """
    labels = np.asarray(etiquetas)
    grupos, inv = np.unique(labels, return_inverse=True)
    k, n = len(grupos), len(labels)

    if k < 2 or k >= n:
        return float("nan")

    es_sparse = sparse.issparse(X)
    A = X.tocsr().astype(np.float64) if es_sparse else np.asarray(X, dtype=float)

    pertenencia = sparse.csr_matrix((np.ones(n), (inv, np.arange(n))), shape=(k, n))
    cuentas = np.bincount(inv, minlength=k).astype(float)

    C = pertenencia @ A
    C = C.toarray() if sparse.issparse(C) else np.asarray(C)
    C = C / cuentas[:, None]

    norma2_x = np.asarray(A.multiply(A).sum(axis=1)).ravel() if es_sparse else np.einsum("ij,ij->i", A, A)
    norma2_c = np.einsum("ij,ij->i", C, C)
    producto = np.asarray(A @ C.T)

    d2 = norma2_x - 2 * producto[np.arange(n), inv] + norma2_c[inv]
    dispersion = np.bincount(inv, weights=np.sqrt(np.maximum(d2, 0.0)), minlength=k) / cuentas

    dist = np.sqrt(np.maximum(norma2_c[:, None] + norma2_c[None, :] - 2 * (C @ C.T), 0.0))
    np.fill_diagonal(dist, np.inf)

    razones = (dispersion[:, None] + dispersion[None, :]) / np.maximum(dist, 1e-12)
    return float(np.mean(np.max(razones, axis=1)))


def adjusted_rand_index(etiquetas_a: np.ndarray, etiquetas_b: np.ndarray) -> float:
    """Índice Rand Ajustado entre dos particiones (1: idénticas, ~0: azar)."""
    a, b = np.asarray(etiquetas_a), np.asarray(etiquetas_b)
    if a.ndim != 1 or b.ndim != 1 or len(a) != len(b):
        raise ValueError("Las dos particiones tienen que ser vectores de la misma longitud.")
    if len(a) < 2:
        return 1.0

    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)

    tabla = np.zeros((ai.max() + 1, bi.max() + 1), dtype=np.int64)
    np.add.at(tabla, (ai, bi), 1)

    comb2 = lambda x: x * (x - 1) / 2.0
    suma_celdas = float(comb2(tabla).sum())
    suma_filas = float(comb2(tabla.sum(axis=1)).sum())
    suma_columnas = float(comb2(tabla.sum(axis=0)).sum())
    total = float(comb2(len(a)))

    esperado = suma_filas * suma_columnas / total
    maximo = 0.5 * (suma_filas + suma_columnas)

    denominador = maximo - esperado
    if denominador == 0:
        return 1.0

    return float((suma_celdas - esperado) / denominador)
