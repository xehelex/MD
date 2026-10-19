"""Métricas internas y específicas para evaluar clustering jerárquico."""
from __future__ import annotations
import numpy as np


def normalizar_filas(X):
    """Devuelve una copia L2 normalizada, conservando dispersión y filas cero."""
    from scipy import sparse
    if sparse.issparse(X):
        A = X.astype(float).tocsr()
        normas = np.sqrt(A.multiply(A).sum(axis=1)).A1
        inversas = np.divide(1.0, normas, out=np.zeros_like(normas), where=normas != 0)
        return A.multiply(inversas[:, None]).tocsr()
    A = np.asarray(X, dtype=float)
    normas = np.linalg.norm(A, axis=1, keepdims=True)
    return np.divide(A, normas, out=np.zeros_like(A), where=normas != 0)


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
