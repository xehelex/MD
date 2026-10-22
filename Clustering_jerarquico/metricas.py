"""Métricas para evaluar una partición.

Calidad de los grupos: silhouette, Davies-Bouldin, Calinski-Harabasz y correlación cofenética.
Estabilidad: ARI, que dice cuánto se parecen dos repartos de los mismos usuarios.
Todas funcionan con X densa (np.ndarray) o dispersa (scipy.sparse).
"""
import numpy as np
from scipy import sparse


# region Funciones auxiliares

def normas(X) -> np.ndarray:
    """Norma L2 de cada fila.

    Pre:  X es una matriz (n, d) densa o dispersa.
    Post: vector de n normas >= 0.
    """
    if sparse.issparse(X):
        return np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    return np.linalg.norm(X, axis=1)


def normalizar_filas(X):
    """Divide cada fila entre su norma para que quede con norma 1.

    Pre:  X es una matriz (n, d) densa o dispersa.
    Post: matriz del mismo tipo y forma. Las filas que eran todo ceros se quedan a cero.
    """
    n = normas(X)
    inversas = np.divide(1.0, n, out=np.zeros_like(n), where=n > 0)  # evita dividir entre 0
    return sparse.diags(inversas) @ X if sparse.issparse(X) else np.asarray(X) * inversas[:, None]


def producto(A, B) -> np.ndarray:
    """Calcula A·Bᵀ y lo devuelve siempre como array denso.

    Pre:  A es (n, d) y B es (m, d), densas o dispersas.
    Post: array (n, m) de float.
    """
    P = A @ B.T
    return np.asarray(P.toarray() if sparse.issparse(P) else P, dtype=float)


def centroides(X, etiquetas):
    """Centroide y tamaño de cada cluster.

    Pre:  X tiene n filas y etiquetas tiene n elementos.
    Post: (C, tam, inv): C es (k, d) con los centroides, tam los tamaños y inv el número
          de cluster (0..k-1) de cada fila. Con X dispersa solo se pasan a denso los k centroides.
    """
    _, inv, tam = np.unique(etiquetas, return_inverse=True, return_counts=True)
    # Matriz (k, n) con un 1 en (cluster, fila): al multiplicarla por X suma las filas de cada cluster
    miembros = sparse.csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))))
    sumas = miembros @ X
    sumas = sumas.toarray() if sparse.issparse(sumas) else np.asarray(sumas)
    return sumas / tam[:, None], tam, inv

# endregion


# region Índices internos

def silhouette_medio(D: np.ndarray, etiquetas) -> float:
    """Silhouette medio: s(i) = (b - a) / max(a, b).

    a = distancia media de i a los de su cluster, b = distancia media al cluster vecino más cercano.

    Pre:  D es la matriz (n, n) de distancias (diagonal a 0) y etiquetas tiene n elementos.
    Post: valor en [-1, 1] (mayor es mejor), o NaN si no hay entre 2 y n-1 clusters.
          Una instancia sola en su cluster cuenta como 0.
    """
    _, inv, tam = np.unique(etiquetas, return_inverse=True, return_counts=True)
    n, k = len(inv), len(tam)
    if not 2 <= k < n:
        return float("nan")
    filas = np.arange(n)
    sumas = D @ np.eye(k)[inv]          # (n, k): suma de distancias de cada instancia a cada cluster
    a = sumas[filas, inv] / np.maximum(tam[inv] - 1, 1)  # sin contarse a sí misma
    medias = sumas / tam
    medias[filas, inv] = np.inf         # para que el mínimo no sea su propio cluster
    b = medias.min(axis=1)
    s = np.divide(b - a, np.maximum(a, b), out=np.zeros(n), where=np.maximum(a, b) > 0)
    s[tam[inv] == 1] = 0.0
    return float(s.mean())


def correlacion_cofenetica(resultado) -> float:
    """Correlación de Pearson entre la distancia original de cada par y la altura a la que el árbol los une.

    Mide lo bien que el dendrograma respeta las distancias de partida.

    Pre:  resultado es un ResultadoJerarquico ya ajustado.
    Post: valor en [-1, 1] (mayor es mejor), o NaN si alguna de las dos series es constante.
    """
    n = resultado.n_instancias
    C = np.zeros((n, n))
    miembros = {i: [i] for i in range(n)}
    # Recorremos las fusiones: todos los pares que se juntan en una fusión tienen esa altura
    for paso, (a, b, altura) in enumerate(resultado.merges):
        ma, mb = miembros.pop(int(a)), miembros.pop(int(b))
        C[np.ix_(ma, mb)] = C[np.ix_(mb, ma)] = altura
        miembros[n + paso] = ma + mb
    D = resultado.distancias_originales
    if resultado.enlace == "ward" and resultado.distancia == "cosine":
        D = np.sqrt(2 * D)              # Ward se construye con la euclídea de las filas normalizadas
    tri = np.triu_indices(n, 1)         # cada par una sola vez
    x, y = D[tri], C[tri]
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def davies_bouldin(X, etiquetas) -> float:
    """Índice de Davies-Bouldin: media, para cada cluster.

    S_i es la distancia media de los puntos del cluster i a su centroide c_i (euclídea).

    Pre:  X tiene n filas (densa o dispersa) y etiquetas tiene n elementos.
    Post: valor >= 0 (menor es mejor), o NaN si no hay entre 2 y n-1 clusters.
    """
    C, tam, inv = centroides(X, etiquetas)
    n, k = len(inv), len(tam)
    if not 2 <= k < n:
        return float("nan")
    cuadrados_c = (C ** 2).sum(axis=1)
    # ||x - c||² = ||x||² - 2·x·c + ||c||², así no hace falta pasar X a densa
    d2 = normas(X) ** 2 - 2 * producto(X, C)[np.arange(n), inv] + cuadrados_c[inv]
    dispersion = np.bincount(inv, weights=np.sqrt(np.maximum(d2, 0))) / tam  # S_i
    entre = np.sqrt(np.maximum(cuadrados_c[:, None] + cuadrados_c[None, :] - 2 * C @ C.T, 0))
    np.fill_diagonal(entre, np.inf)     # un cluster no se compara consigo mismo
    razones = (dispersion[:, None] + dispersion[None, :]) / np.maximum(entre, 1e-12)
    return float(razones.max(axis=1).mean())


def calinski_harabasz(X, etiquetas) -> float:
    """Índice de Calinski-Harabasz: (B / (k - 1)) / (W / (n - k)).

    B es la dispersión entre clusters y W la dispersión dentro de los clusters.

    Pre:  X tiene n filas (densa o dispersa) y etiquetas tiene n elementos.
    Post: valor >= 0 (mayor es mejor), o NaN si no hay entre 2 y n-1 clusters o W = 0.
    """
    C, tam, inv = centroides(X, etiquetas)
    n, k = len(inv), len(tam)
    if not 2 <= k < n:
        return float("nan")
    media = tam @ C / n                 # centroide global
    entre = float((tam * ((C - media) ** 2).sum(axis=1)).sum())
    # W = suma de ||x||² - suma de tam·||c||² (misma idea que en Davies-Bouldin)
    dentro = float((normas(X) ** 2).sum() - (tam * (C ** 2).sum(axis=1)).sum())
    if dentro <= 1e-12:
        return float("nan")
    return (entre / (k - 1)) / (dentro / (n - k))

# endregion


# region Estabilidad

def adjusted_rand_index(reparto_a, reparto_b) -> float:
    """Índice de Rand ajustado (ARI): cuánto se parecen dos repartos en grupos de los mismos usuarios.

    Mira cada par de usuarios y comprueba si los dos repartos coinciden en ponerlos juntos o separados,
    descontando lo que coincidirían por azar.

    Pre:  los dos repartos tienen la misma longitud y la posición i es el mismo usuario en los dos.
    Post: 1 si son iguales, alrededor de 0 si solo coinciden por azar (puede ser negativo).
    """
    _, a = np.unique(reparto_a, return_inverse=True)
    _, b = np.unique(reparto_b, return_inverse=True)
    if len(a) != len(b):
        raise ValueError("Los dos repartos tienen que tener la misma longitud.")
    if len(a) < 2:
        return 1.0
    # Tabla de contingencia: cuántos usuarios están en el grupo i del reparto A y en el j del B
    tabla = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(tabla, (a, b), 1)
    pares = lambda x: float((x * (x - 1) / 2).sum())   # suma de "x sobre 2"
    celdas, filas, columnas = pares(tabla), pares(tabla.sum(axis=1)), pares(tabla.sum(axis=0))
    esperado = filas * columnas / (len(a) * (len(a) - 1) / 2)
    maximo = (filas + columnas) / 2
    return 1.0 if maximo == esperado else (celdas - esperado) / (maximo - esperado)

# endregion


def evaluar_corte(resultado, X, k: int) -> dict:
    """Corta el árbol en k clusters y calcula sus métricas.

    Pre:  resultado es el árbol ajustado con X y 2 <= k < n.
    Post: diccionario con k, silhouette, davies_bouldin, calinski_harabasz, tam_min, tam_max
          y corte_ambiguo.
    """
    etiquetas = resultado.etiquetas(k)
    tam = np.bincount(etiquetas)
    return {
        "k": k,
        "silhouette": silhouette_medio(resultado.distancias_originales, etiquetas),
        "davies_bouldin": davies_bouldin(X, etiquetas),
        "calinski_harabasz": calinski_harabasz(X, etiquetas),
        "tam_min": int(tam.min()),
        "tam_max": int(tam.max()),
        "corte_ambiguo": resultado.corte_ambiguo(k),
    }
