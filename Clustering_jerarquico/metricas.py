"""Métricas para evaluar una partición: silhouette, cofenética, Davies-Bouldin, Calinski-Harabasz y ARI."""
from __future__ import annotations
import numpy as np
from scipy import sparse


def silhouette_medio(D: np.ndarray, etiquetas: np.ndarray) -> float:
    """Silhouette medio de una partición a partir de su matriz de distancias.

    Pre:  D es cuadrada (n, n) y etiquetas tiene n elementos.
    Post: devuelve un valor en [-1, 1], o NaN si hay menos de dos clusters o uno por instancia.
          Una instancia sola en su cluster puntúa 0.
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
            continue

        a = float(D[i, mismo].mean())
        b = min(float(D[i, labels == g].mean()) for g in grupos if g != labels[i])
        denom = max(a, b)
        valores[i] = 0.0 if denom == 0 else (b - a) / denom

    return float(np.mean(valores))


def correlacion_cofenetica(resultado) -> float:
    """Correlación de Pearson entre las distancias originales y las alturas a las que el árbol une cada par.

    Pre:  resultado es un ResultadoJerarquico ya ajustado.
    Post: devuelve un valor en [-1, 1], o NaN si alguna de las dos series es constante.
    """
    D = resultado.distancias_originales
    n = resultado.n_instancias
    cofenetica = np.zeros((n, n), dtype=float)

    miembros = {i: [i] for i in range(n)}
    for idx, (izquierda, derecha, altura) in enumerate(resultado.merges):
        mi, mj = miembros.pop(int(izquierda)), miembros.pop(int(derecha))
        for a in mi:
            for b in mj:
                cofenetica[a, b] = cofenetica[b, a] = altura
        miembros[n + idx] = mi + mj

    tri = np.triu_indices(n, 1)
    x, y = D[tri], cofenetica[tri]

    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")

    return float(np.corrcoef(x, y)[0, 1])


def davies_bouldin(X, etiquetas: np.ndarray) -> float:
    """Índice de Davies-Bouldin con distancia euclídea entre centroides (menor es mejor).

    Pre:  X tiene n filas (densa o dispersa) y etiquetas tiene n elementos.
    Post: devuelve un valor >= 0, o NaN si hay menos de dos clusters o uno por instancia.
    Una matriz dispersa se pasa a densa, así que conviene usarlo con conjuntos pequeños o medianos.
    """
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
    """Índice de Rand ajustado (ARI) entre dos particiones de las mismas instancias.

    Pre:  las dos etiquetas son vectores de la misma longitud.
    Post: devuelve 1 si las particiones son iguales y un valor cercano a 0 si solo coinciden por azar.
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

    def pares(x):
        return x * (x - 1) / 2.0

    suma_celdas = float(pares(tabla).sum())
    suma_filas = float(pares(tabla.sum(axis=1)).sum())
    suma_columnas = float(pares(tabla.sum(axis=0)).sum())
    total = float(pares(len(a)))

    esperado = suma_filas * suma_columnas / total
    maximo = 0.5 * (suma_filas + suma_columnas)

    denominador = maximo - esperado
    if denominador == 0:
        return 1.0

    return float((suma_celdas - esperado) / denominador)


def normalizar_filas(X):
    """Escala cada fila a norma L2 igual a 1.

    Pre:  X es una matriz (n, d) densa o dispersa.
    Post: devuelve una matriz del mismo tipo y forma; las filas nulas se quedan en cero.
    """
    if sparse.issparse(X):
        normas = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
        inversas = np.zeros_like(normas, dtype=float)
        np.divide(1.0, normas, out=inversas, where=normas != 0)
        return sparse.diags(inversas) @ X

    X = np.asarray(X, dtype=float)
    normas = np.linalg.norm(X, axis=1, keepdims=True)
    return np.divide(X, normas, out=np.zeros_like(X), where=normas != 0)


def calinski_harabasz(X, etiquetas: np.ndarray) -> float:
    """Índice de Calinski-Harabasz: (B / (k - 1)) / (W / (n - k)), con B y W las dispersiones entre y dentro de clusters.

    Pre:  X tiene n filas (densa o dispersa) y etiquetas tiene n elementos.
    Post: devuelve un valor >= 0 (mayor es mejor), o NaN si hay menos de dos clusters, uno por instancia o W = 0.
    No densifica X: con matrices dispersas solo se construyen los k centroides.
    """
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
