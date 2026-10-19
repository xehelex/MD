"""Métricas internas y específicas para evaluar clustering jerárquico."""
from __future__ import annotations
import numpy as np


def silhouette_medio(D: np.ndarray, etiquetas: np.ndarray) -> float:
    """Calcula el silhouette medio pasándole una matriz de distancias ya calculada.

    Devuelve NaN si hay menos de dos clusters o si cada instancia está en su
    propio cluster. Si un punto está solito en un cluster, le ponemos 0.
    """
    D = np.asarray(D, dtype=float)
    labels = np.asarray(etiquetas)
    n = len(labels)

    if D.shape != (n, n):
        raise ValueError("D tiene que ser de tamaño (n, n) y encajar con las etiquetas.")

    grupos = np.unique(labels)
    if len(grupos) < 2 or len(grupos) >= n:
        return float("nan")

    valores = np.zeros(n, dtype=float)
    for i in range(n):
        mismo = labels == labels[i]
        mismo[i] = False
        if not np.any(mismo):
            valores[i] = 0.0
            continue

        a = float(D[i, mismo].mean())
        b = min(float(D[i, labels == g].mean()) for g in grupos if g != labels[i])
        denom = max(a, b)
        valores[i] = 0.0 if denom == 0 else (b - a) / denom

    return float(np.mean(valores))


def correlacion_cofenetica(resultado) -> float:
    """Mide la correlación de Pearson entre las distancias originales y las alturas del árbol."""
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


def davies_bouldin(X, etiquetas: np.ndarray) -> float:
    """Calcula el índice Davies-Bouldin en su versión euclídea (cuanto menor, mejor).

    Va bien tenerlo como métrica extra, aunque ojo: para matrices dispersas
    toca densificar los datos, así que úsalo solo con datasets pequeños o medianos.
    """
    from scipy import sparse
    A = X.toarray() if sparse.issparse(X) else np.asarray(X, dtype=float)
    labels = np.asarray(etiquetas)
    grupos = np.unique(labels)

    if len(grupos) < 2 or len(grupos) >= len(labels):
        return float("nan")

    centroides, dispersion = [], []
    for g in grupos:
        puntos = A[labels == g]
        centro = puntos.mean(axis=0)
        centroides.append(centro)
        dispersion.append(float(np.linalg.norm(puntos - centro, axis=1).mean()))

    centroides = np.asarray(centroides)
    dispersion = np.asarray(dispersion)
    dist = np.linalg.norm(centroides[:, None, :] - centroides[None, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)

    razones = (dispersion[:, None] + dispersion[None, :]) / np.maximum(dist, 1e-12)
    return float(np.mean(np.max(razones, axis=1)))


def adjusted_rand_index(etiquetas_a: np.ndarray, etiquetas_b: np.ndarray) -> float:
    """Calcula el Índice Rand Ajustado (ARI) entre dos particiones sin depender de sklearn.

    Ideal para comparar contra una referencia real o para ver qué tan estables
    son dos ejecuciones distintas sobre las mismas filas.
    - Cerca de 1: son prácticamente iguales.
    - Alrededor de 0: coinciden lo esperable por pura chiripa (azar).
    """
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

def normalizar_filas(X):
    """Normaliza cada fila a norma L2 igual a 1, sin usar sklearn."""
    from scipy import sparse

    if sparse.issparse(X):
        normas = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
        inversas = np.zeros_like(normas, dtype=float)
        np.divide(1.0, normas, out=inversas, where=normas != 0)
        return sparse.diags(inversas) @ X

    X = np.asarray(X, dtype=float)
    normas = np.linalg.norm(X, axis=1, keepdims=True)
    return np.divide(X, normas, out=np.zeros_like(X), where=normas != 0)


def calinski_harabasz(X, etiquetas: np.ndarray) -> float:
    """Calcula el índice de Calinski-Harabasz (cuanto mayor, mejor).

    Cociente entre la dispersión entre clusters (B) y la dispersión interna (W),
    ponderadas por sus grados de libertad: (B / (k - 1)) / (W / (n - k)).
    No densifica X: con matrices dispersas solo se materializan los k centroides.
    Devuelve NaN si hay menos de dos clusters, si cada instancia está sola o si W = 0.
    """
    from scipy import sparse

    labels = np.asarray(etiquetas)
    n = len(labels)
    grupos, inv = np.unique(labels, return_inverse=True)
    k = len(grupos)
    if k < 2 or k >= n:
        return float("nan")

    es_sparse = sparse.issparse(X)
    A = X.tocsr().astype(np.float64) if es_sparse else np.asarray(X, dtype=float)
    if A.shape[0] != n:
        raise ValueError("X y las etiquetas tienen que tener el mismo número de filas.")

    miembros = sparse.csr_matrix((np.ones(n), (inv, np.arange(n))), shape=(k, n))
    sumas = miembros @ A
    sumas = sumas.toarray() if es_sparse else np.asarray(sumas)

    tam = np.bincount(inv).astype(float)
    centroides = sumas / tam[:, None]
    media = np.asarray(A.sum(axis=0)).ravel() / n

    entre = float((tam * ((centroides - media) ** 2).sum(axis=1)).sum())
    total = float(A.multiply(A).sum()) if es_sparse else float((A * A).sum())
    dentro = max(total - float((tam * (centroides ** 2).sum(axis=1)).sum()), 0.0)

    if dentro <= 1e-12:
        return float("nan")
    return float((entre / (k - 1)) / (dentro / (n - k)))
