from __future__ import annotations
import numpy as np


def silhouette_medio(D: np.ndarray, etiquetas: np.ndarray) -> float:

    """
    D = np.asarray(D, dtype=float)
    labels = np.asarray(etiquetas)
    n = len(labels)

    if D.shape != (n, n):
        raise ValueError("D tiene que ser de tamaño (n, n) y encajar con las etiquetas.")

        return float("nan")





def correlacion_cofenetica(resultado) -> float:
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

    """
    labels = np.asarray(etiquetas)

        return float("nan")


    np.fill_diagonal(dist, np.inf)

    razones = (dispersion[:, None] + dispersion[None, :]) / np.maximum(dist, 1e-12)
    return float(np.mean(np.max(razones, axis=1)))


def adjusted_rand_index(etiquetas_a: np.ndarray, etiquetas_b: np.ndarray) -> float:
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