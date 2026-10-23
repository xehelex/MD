"""Clustering jerárquico aglomerativo (implementación propia) y el modelo final que se guarda en modelo.pkl.

Flujo: distancias() -> ClusteringJerarquico.fit() -> ResultadoJerarquico (árbol) -> etiquetas(k) (corte)
       entrenar_modelo() junta todo eso y devuelve un ModeloEntrenado que sabe predecir filas nuevas.
"""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import pickle
import numpy as np
from scipy import sparse
from .metricas import normas, normalizar_filas, producto, centroides, evaluar_corte, correlacion_cofenetica

ENLACES = ("single", "complete", "average", "ward")
DISTANCIAS = ("cosine", "euclidean", "manhattan", "chebyshev", "minkowski")


def distancias(A, B, tipo: str = "cosine", p: float = 2.0) -> np.ndarray:
    """Distancia entre cada fila de A y cada fila de B.

    Pre:  A es (n, d) y B es (m, d), densas o dispersas; tipo está en DISTANCIAS; p >= 1 (solo minkowski).
    Post: array (n, m) con las distancias (coseno en [0, 2], el resto >= 0).
    """
    if tipo == "minkowski" and p == 2:
        tipo = "euclidean"

    # Coseno y euclídea salen del producto escalar, que funciona igual con matrices dispersas
    if tipo in ("cosine", "euclidean"):
        P, na, nb = producto(A, B), normas(A), normas(B)
        if tipo == "euclidean":
            # ||a - b||² = ||a||² + ||b||² - 2·a·b  (el maximum quita negativos por redondeo)
            return np.sqrt(np.maximum(na[:, None] ** 2 + nb[None, :] ** 2 - 2 * P, 0.0))
        denom = np.outer(na, nb)
        D = 1.0 - np.divide(P, denom, out=np.zeros_like(P), where=denom > 0)
        D[np.ix_(na == 0, nb == 0)] = 0.0      # dos vectores nulos los consideramos iguales
        return np.clip(D, 0.0, 2.0)

    # Manhattan (p = 1), Chebyshev (p = inf) y Minkowski: hay que pasar a densa e ir fila a fila, O(n·m·d)
    p = {"manhattan": 1.0, "chebyshev": np.inf}.get(tipo, p)
    A = A.toarray() if sparse.issparse(A) else np.asarray(A, dtype=float)
    B = B.toarray() if sparse.issparse(B) else np.asarray(B, dtype=float)
    D = np.empty((len(A), len(B)))
    for i, fila in enumerate(A):
        dif = np.abs(B - fila)
        D[i] = dif.max(axis=1) if np.isinf(p) else (dif ** p).sum(axis=1) ** (1 / p)
    return D


@dataclass
class ResultadoJerarquico:
    """Árbol de fusiones que devuelve fit().

    Cada fila de `merges` es (id_a, id_b, altura). Las hojas son los ids 0..n-1 y la fusión
    número i crea el nodo n+i (mismo formato que usa scipy, para poder comparar).
    """
    merges: np.ndarray
    n_instancias: int
    enlace: str
    distancia: str
    distancias_originales: np.ndarray       # matriz (n, n), se usa para el silhouette y la cofenética

    def etiquetas(self, k: int) -> np.ndarray:
        """Corta el árbol en k clusters.

        Pre:  1 <= k <= n_instancias.
        Post: vector de n enteros con k valores distintos (0..k-1).
        """
        n = self.n_instancias
        if not 1 <= k <= n:
            raise ValueError(f"k tiene que estar entre 1 y {n}.")
        # Para quedarnos con k clusters basta con aplicar las n-k primeras fusiones
        grupos = {i: [i] for i in range(n)}
        for paso, (a, b, _) in enumerate(self.merges[:n - k]):
            grupos[n + paso] = grupos.pop(int(a)) + grupos.pop(int(b))
        etiquetas = np.empty(n, dtype=int)
        for etiqueta, miembros in enumerate(grupos.values()):
            etiquetas[miembros] = etiqueta
        return etiquetas

    def corte_ambiguo(self, k: int) -> bool:
        """Dice si el corte en k depende del desempate.

        Pre:  k es un entero.
        Post: True si las fusiones justo antes y justo después del corte tienen la misma altura
              (pasa mucho con complete sobre TF-IDF, donde muchas distancias valen exactamente 1).
        """
        h, n = self.merges[:, 2], self.n_instancias
        return bool(2 <= k < n and abs(h[n - k - 1] - h[n - k]) <= 1e-12)


class ClusteringJerarquico:
    """Clustering aglomerativo con enlaces single, complete, average y ward.

    Coste: memoria O(n²) por la matriz de distancias y tiempo O(n³) (n-1 pasos que buscan
    el mínimo en una matriz n x n). Con nuestros ~600 usuarios tarda menos de medio segundo.
    """

    def __init__(self, enlace: str = "average", distancia: str = "cosine", p: float = 2.0):
        """Pre:  enlace en ENLACES, distancia en DISTANCIAS y p >= 1 si la distancia es minkowski.
                 Ward solo con euclídea (o minkowski p = 2) o coseno.
        Post: objeto listo para fit(). Si no se cumple la precondición lanza ValueError.
        """
        if enlace not in ENLACES:
            raise ValueError(f"Enlace '{enlace}' no válido. Usa: {ENLACES}")
        if distancia not in DISTANCIAS:
            raise ValueError(f"Distancia '{distancia}' no válida. Usa: {DISTANCIAS}")
        if distancia == "minkowski" and p < 1:
            raise ValueError("El p de Minkowski tiene que ser >= 1.")
        if enlace == "ward" and distancia not in ("euclidean", "cosine") and not (distancia == "minkowski" and p == 2):
            raise ValueError("Ward solo admite distancia euclídea o coseno.")
        self.enlace, self.distancia, self.p = enlace, distancia, float(p)

    def fit(self, X) -> ResultadoJerarquico:
        """Construye el árbol de fusiones.

        Pre:  X es una matriz (n, d) densa o dispersa con n >= 2 y valores finitos.
        Post: ResultadoJerarquico con las n-1 fusiones en el orden en que ocurren.
        """
        # 1. Matriz de distancias inicial
        X = X.astype(np.float64)                # nuestras matrices vienen en float32 y se pierde precisión
        D0 = np.triu(distancias(X, X, self.distancia, self.p), 1)
        D0 = D0 + D0.T                          # simétrica del todo y con la diagonal a 0
        n = len(D0)
        if n < 2:
            raise ValueError("Hacen falta al menos 2 instancias.")

        # 2. Matriz de trabajo. Ward va con distancias euclídeas al cuadrado.
        #    Con coseno usamos que en vectores de norma 1: d_euclídea² = 2 · d_coseno
        ward = self.enlace == "ward"
        if ward:
            D = 2 * D0 if self.distancia == "cosine" else D0 ** 2
        else:
            D = D0.copy()
        np.fill_diagonal(D, np.inf)             # que nadie se fusione consigo mismo
        tam = np.ones(n)                        # tamaño del cluster de cada fila
        nodo = np.arange(n)                     # id del cluster que ocupa cada fila de D
        merges = []

        # 3. n-1 fusiones: en cada paso se juntan los dos clusters más cercanos
        for paso in range(n - 1):
            i, j = sorted(np.unravel_index(np.argmin(D), D.shape))
            altura = D[i, j]
            merges.append((nodo[i], nodo[j], altura))

            # Distancia del cluster nuevo (i ∪ j) a todos los demás (fórmulas de Lance-Williams)
            if self.enlace == "single":
                nuevo = np.minimum(D[i], D[j])
            elif self.enlace == "complete":
                nuevo = np.maximum(D[i], D[j])
            elif self.enlace == "average":
                nuevo = (tam[i] * D[i] + tam[j] * D[j]) / (tam[i] + tam[j])
            else:  # ward
                nuevo = ((tam[i] + tam) * D[i] + (tam[j] + tam) * D[j] - tam * altura) / (tam[i] + tam[j] + tam)

            # El cluster nuevo se queda en la fila i y la fila j se "apaga" poniéndola a infinito
            D[i, :] = D[:, i] = nuevo
            D[j, :] = D[:, j] = np.inf
            D[i, i] = np.inf
            tam[i] += tam[j]
            nodo[i] = n + paso

        merges = np.array(merges, dtype=float)
        if ward:
            merges[:, 2] = np.sqrt(merges[:, 2])  # volvemos a unidades de distancia
        return ResultadoJerarquico(merges, n, self.enlace, self.distancia, D0)

    def fit_predict(self, X, k: int) -> np.ndarray:
        """Pre: como fit() y 1 <= k <= n.  Post: etiquetas del corte en k clusters."""
        return self.fit(X).etiquetas(k)


@dataclass
class ModeloEntrenado:
    """Modelo final que se guarda en modelo.pkl.

    Guarda el árbol, la matriz de entrenamiento, las etiquetas y los centroides para poder asignar
    usuarios nuevos. Para cargarlo el paquete Clustering_jerarquico tiene que poder importarse
    (pickle guarda la clase por su nombre) y solo hay que abrir .pkl de confianza.
    """
    enlace: str
    distancia: str
    k: int
    resultado: ResultadoJerarquico
    X: object                  # matriz de entrenamiento (normalizada por filas si la distancia es coseno)
    etiquetas: np.ndarray
    centroides: np.ndarray
    tamanos: np.ndarray
    metricas: dict
    creado: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def predecir(self, X_nuevo) -> np.ndarray:
        """Asigna un cluster a cada fila nueva.

        El jerárquico no predice de por sí, así que cada fila va al cluster al que menos le costaría
        unirse según el enlace: single = distancia mínima, complete = máxima, average = media y
        ward = lo que aumenta la suma de cuadrados.
        Es una aproximación: el árbol no se recalcula.

        Pre:  X_nuevo (una fila o una matriz) tiene los mismos atributos que el entrenamiento.
        Post: vector con un cluster (0..k-1) por fila.
        """
        if not sparse.issparse(X_nuevo):
            X_nuevo = np.atleast_2d(np.asarray(X_nuevo, dtype=float))  # admite también una sola fila
        if X_nuevo.shape[1] != self.X.shape[1]:
            raise ValueError(f"Las filas nuevas tienen que tener {self.X.shape[1]} atributos.")
        if self.distancia == "cosine":
            X_nuevo = normalizar_filas(X_nuevo)   # igual que se hizo con el entrenamiento
        if self.enlace == "ward":
            coste = distancias(X_nuevo, self.centroides, "euclidean") ** 2 * self.tamanos / (self.tamanos + 1)
        else:
            D = distancias(X_nuevo, self.X, self.distancia)
            agregar = {"single": np.min, "complete": np.max, "average": np.mean}[self.enlace]
            coste = np.column_stack([agregar(D[:, self.etiquetas == c], axis=1) for c in range(self.k)])
        return np.argmin(coste, axis=1)

    def resumen(self) -> str:
        """Pre:  el modelo viene de entrenar_modelo (metricas tiene silhouette, davies_bouldin,
              calinski_harabasz y cofenetica).
        Post: texto con la configuración, los tamaños de cluster y las métricas del modelo."""
        m = self.metricas
        return (f"Enlace: {self.enlace} | distancia: {self.distancia} | k = {self.k}\n"
                f"Instancias: {self.X.shape[0]} | atributos: {self.X.shape[1]}\n"
                f"Tamaños de cluster: {self.tamanos.tolist()}\n"
                f"Silhouette: {m['silhouette']:.4f} | Davies-Bouldin: {m['davies_bouldin']:.4f} | "
                f"Calinski-Harabasz: {m['calinski_harabasz']:.2f} | Cofenética: {m['cofenetica']:.4f}")

    def guardar(self, ruta) -> Path:
        """Pre: ruta es la ruta del .pkl.  Post: modelo guardado (crea las carpetas que falten); devuelve la ruta."""
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        with ruta.open("wb") as f:
            pickle.dump(self, f)
        return ruta

    @staticmethod
    def cargar(ruta) -> "ModeloEntrenado":
        """Pre: ruta apunta a un .pkl creado con guardar().  Post: el ModeloEntrenado guardado."""
        with Path(ruta).open("rb") as f:
            return pickle.load(f)


def entrenar_modelo(X, enlace: str, distancia: str, k: int) -> ModeloEntrenado:
    """Entrena el modelo final con la configuración elegida.

    Pre:  X es (n, d) densa o dispersa, distancia es 'cosine' o 'euclidean' y 2 <= k < n.
    Post: ModeloEntrenado con el árbol cortado en k clusters, sus centroides y sus métricas.
    """
    if distancia not in ("cosine", "euclidean"):
        raise ValueError("Para el modelo final la distancia tiene que ser 'cosine' o 'euclidean'.")
    # Con coseno normalizamos: las distancias no cambian y los centroides (Ward, DB, CH) tienen sentido
    X = normalizar_filas(X) if distancia == "cosine" else X
    resultado = ClusteringJerarquico(enlace, distancia).fit(X)
    if not 2 <= k < resultado.n_instancias:
        raise ValueError(f"k tiene que cumplir 2 <= k < {resultado.n_instancias}.")
    etiquetas = resultado.etiquetas(k)
    C, tam, _ = centroides(X, etiquetas)
    metricas = {**evaluar_corte(resultado, X, k), "cofenetica": correlacion_cofenetica(resultado)}
    return ModeloEntrenado(enlace, distancia, k, resultado, X, etiquetas, C, tam, metricas)
