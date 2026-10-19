"""Clustering jerárquico aglomerativo propio y el modelo final que se guarda en modelo.pkl."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import heapq
import itertools
import pickle
from pathlib import Path
import numpy as np
from scipy import sparse
from .metricas import (silhouette_medio, correlacion_cofenetica, davies_bouldin,
                       calinski_harabasz, normalizar_filas)

# Distancias admitidas para el modelo final (modelo.pkl).
DISTANCIAS_MODELO = ("cosine", "euclidean")


@dataclass
class ResultadoJerarquico:
    """Árbol de fusiones: cada fila de `merges` es (id_izquierda, id_derecha, altura).

    Las hojas son los ids 0..n-1 y la fusión número i crea el nodo n+i.
    """
    merges: np.ndarray
    n_instancias: int
    enlace: str
    distancia: str
    distancias_originales: np.ndarray

    def etiquetas(self, k: int) -> np.ndarray:
        """Corta el árbol en k clusters.

        Pre:  1 <= k <= n_instancias.
        Post: vector de n enteros con exactamente k valores distintos.
        """
        n = self.n_instancias
        if not 1 <= k <= n:
            raise ValueError(f"k tiene que estar entre 1 y {n}.")

        grupos = {i: {i} for i in range(n)}
        siguiente_id = n

        for izquierda, derecha, _altura in self.merges[:n - k]:
            izquierda, derecha = int(izquierda), int(derecha)
            grupos[siguiente_id] = grupos.pop(izquierda) | grupos.pop(derecha)
            siguiente_id += 1

        etiquetas = np.empty(n, dtype=int)
        for etiqueta, miembros in enumerate(grupos.values()):
            for miembro in miembros:
                etiquetas[miembro] = etiqueta
        return etiquetas

    def asignar_nuevo(self, distancia_nuevo: np.ndarray, k: int) -> int:
        """Asigna un elemento nuevo al cluster con menor distancia media (aproximación, el árbol no cambia).

        Pre:  distancia_nuevo tiene una distancia finita a cada instancia de entrenamiento.
        Post: etiqueta de uno de los k clusters.
        """
        d = np.asarray(distancia_nuevo, dtype=float)
        if d.shape != (self.n_instancias,):
            raise ValueError("distancia_nuevo debe tener una distancia por cada instancia.")
        if not np.all(np.isfinite(d)):
            raise ValueError("Las distancias tienen que ser valores finitos.")

        labels = self.etiquetas(k)
        candidatos = np.unique(labels)
        medias = [float(d[labels == c].mean()) for c in candidatos]
        return int(candidatos[int(np.argmin(medias))])


class ClusteringJerarquico:
    """Clustering aglomerativo con enlaces single, complete, average y ward.

    Construye la matriz de distancias completa, así que la memoria crece como O(n²).
    Manhattan, Chebyshev y Minkowski con p != 2 convierten X a densa y cuestan O(n²·d).
    """
    ENLACES = {"single", "complete", "average", "ward"}
    DISTANCIAS = {"cosine", "euclidean", "manhattan", "chebyshev", "minkowski"}

    def __init__(self, enlace: str = "average", distancia: str = "cosine", p: float = 2.0):
        """Pre:  enlace y distancia pertenecen a ENLACES y DISTANCIAS; p >= 1 si la distancia es minkowski.

        Lanza ValueError si no se cumple. Ward solo admite distancia euclídea (minkowski con p = 2 incluida).
        """
        if enlace not in self.ENLACES:
            raise ValueError(f"Enlace '{enlace}' no válido. Usa: {sorted(self.ENLACES)}")
        if distancia not in self.DISTANCIAS:
            raise ValueError(f"Distancia '{distancia}' no válida. Usa: {sorted(self.DISTANCIAS)}")
        if distancia == "minkowski" and not (p >= 1):
            raise ValueError("El p de Minkowski tiene que ser >= 1 (admite np.inf).")

        if enlace == "ward":
            euclidea = distancia == "euclidean" or (distancia == "minkowski" and p == 2)
            if not euclidea:
                raise ValueError("Ward solo tiene sentido con distancia euclídea (minkowski p=2).")

        self.enlace = enlace
        self.distancia = distancia
        self.p = float(p)
        self.resultado_: ResultadoJerarquico | None = None

    @staticmethod
    def _minkowski(A: np.ndarray, p: float) -> np.ndarray:
        """Distancias de Minkowski calculadas por bloques de filas (unos 256 MB por bloque)."""
        n, d = A.shape
        bloque = max(1, int(256e6 // max(n * d * 8, 1)))
        D = np.zeros((n, n), dtype=float)

        for ini in range(0, n, bloque):
            fin = min(ini + bloque, n)
            dif = np.abs(A[ini:fin, None, :] - A[None, :, :])
            if np.isinf(p):
                D[ini:fin] = dif.max(axis=2)
            elif p == 1:
                D[ini:fin] = dif.sum(axis=2)
            else:
                D[ini:fin] = (dif ** p).sum(axis=2) ** (1.0 / p)

        return D

    @staticmethod
    def _distancias(X, tipo: str, p: float = 2.0) -> np.ndarray:
        """Matriz de distancias (n, n) de las filas de X, que puede ser densa o dispersa."""
        if X.ndim != 2:
            raise ValueError("X tiene que ser una matriz de 2 dimensiones.")
        n = X.shape[0]
        if n < 2:
            raise ValueError("Hacen falta al menos 2 instancias.")

        exponentes = {"manhattan": 1.0, "chebyshev": np.inf, "minkowski": p}
        if tipo in exponentes and not (tipo == "minkowski" and p == 2):
            A = X.toarray() if sparse.issparse(X) else np.asarray(X, dtype=float)
            if not np.all(np.isfinite(A)):
                raise ValueError("X tiene valores NaN o infinitos.")
            D = ClusteringJerarquico._minkowski(A, exponentes[tipo])
            np.fill_diagonal(D, 0.0)
            return D

        if sparse.issparse(X):
            X = X.astype(np.float64, copy=False)
            normas = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
            producto = (X @ X.T).toarray()
            if tipo == "cosine":
                denom = np.outer(normas, normas)
                similitud = np.divide(producto, denom, out=np.zeros_like(producto), where=denom > 0)
                D = np.clip(1.0 - similitud, 0.0, 2.0)
                ceros = normas == 0
                D[np.ix_(ceros, ceros)] = 0.0
            else:
                norma2 = normas ** 2
                D = np.sqrt(np.maximum(norma2[:, None] + norma2[None, :] - 2 * producto, 0.0))
        else:
            A = np.asarray(X, dtype=float)
            if not np.all(np.isfinite(A)):
                raise ValueError("X tiene valores NaN o infinitos.")
            if tipo == "cosine":
                normas = np.linalg.norm(A, axis=1)
                producto = A @ A.T
                denom = np.outer(normas, normas)
                similitud = np.divide(producto, denom, out=np.zeros_like(producto), where=denom > 0)
                D = np.clip(1.0 - similitud, 0.0, 2.0)
                ceros = normas == 0
                D[np.ix_(ceros, ceros)] = 0.0
            else:
                norma2 = np.sum(A * A, axis=1)
                D = np.sqrt(np.maximum(norma2[:, None] + norma2[None, :] - 2 * (A @ A.T), 0.0))

        np.fill_diagonal(D, 0.0)
        return D

    def fit(self, X) -> ResultadoJerarquico:
        """Construye el árbol de fusiones.

        Pre:  X tiene n >= 2 filas, es finita y puede ser densa o dispersa.
        Post: devuelve (y guarda en `resultado_`) un ResultadoJerarquico con n-1 fusiones,
              en el orden en que se producen (altura no decreciente con estos cuatro enlaces).
        """
        D0 = self._distancias(X, self.distancia, self.p)
        n = D0.shape[0]

        # Ward se actualiza sobre distancias al cuadrado; D0 se conserva para las métricas.
        es_ward = self.enlace == "ward"
        D_trabajo = D0 ** 2 if es_ward else D0

        dist = {}
        for i in range(n):
            for j in range(i + 1, n):
                dist[(i, j)] = float(D_trabajo[i, j])

        sizes = {i: 1 for i in range(n)}
        active = set(range(n))
        heap = []
        counter = itertools.count()

        for (i, j), d in dist.items():
            heapq.heappush(heap, (d, next(counter), i, j))

        merges = []
        next_id = n

        for _ in range(n - 1):
            # La cola acumula entradas obsoletas: se descartan al sacarlas si algún cluster
            # ya no está activo o si la distancia guardada ha cambiado.
            while heap:
                d, _seq, i, j = heapq.heappop(heap)
                if i in active and j in active and (min(i, j), max(i, j)) in dist:
                    actual = dist[(min(i, j), max(i, j))]
                    if abs(actual - d) <= 1e-12:
                        break
            else:
                raise RuntimeError("No se ha podido encontrar una fusión válida.")

            merges.append((i, j, d))
            nuevos = next_id
            next_id += 1
            otros = list(active - {i, j})
            size_i, size_j = sizes[i], sizes[j]

            # Fórmulas de Lance-Williams para la distancia del cluster nuevo al resto.
            for z in otros:
                diz = dist.pop((min(i, z), max(i, z)))
                djz = dist.pop((min(j, z), max(j, z)))

                if self.enlace == "single":
                    dnz = min(diz, djz)
                elif self.enlace == "complete":
                    dnz = max(diz, djz)
                elif self.enlace == "ward":
                    size_z = sizes[z]
                    dnz = ((size_i + size_z) * diz + (size_j + size_z) * djz
                           - size_z * d) / (size_i + size_j + size_z)
                else:
                    dnz = (size_i * diz + size_j * djz) / (size_i + size_j)

                key = (min(nuevos, z), max(nuevos, z))
                dist[key] = float(dnz)
                heapq.heappush(heap, (float(dnz), next(counter), key[0], key[1]))

            dist.pop((min(i, j), max(i, j)))
            active.remove(i)
            active.remove(j)
            active.add(nuevos)
            sizes[nuevos] = size_i + size_j

        merges = np.asarray(merges, dtype=float)
        if es_ward and len(merges):
            # Las alturas de Ward vuelven a unidades de distancia (raíz de la altura cuadrática).
            merges[:, 2] = np.sqrt(np.maximum(merges[:, 2], 0.0))

        self.resultado_ = ResultadoJerarquico(
            merges=merges,
            n_instancias=n,
            enlace=self.enlace,
            distancia=self.distancia,
            distancias_originales=D0,
        )
        return self.resultado_

    def fit_predict(self, X, k: int) -> np.ndarray:
        """Ajusta el modelo y devuelve las etiquetas de un corte en k clusters (Pre/Post como `fit` y `etiquetas`)."""
        return self.fit(X).etiquetas(k)


def preparar_modelo(enlace: str, distancia: str, X, p: float = 2.0):
    """Crea el clustering y devuelve (modelo, matriz con la que hay que ajustarlo).

    Ward solo admite distancia euclídea. Si se pide coseno con Ward, se normalizan las filas a norma 1
    y se usa la euclídea: sobre vectores unitarios d_eucl² = 2·d_coseno, así que Ward sigue
    trabajando en el espacio coseno. Con el resto de combinaciones X no se modifica.

    Lanza ValueError si la combinación enlace-distancia no es válida.
    """
    if enlace == "ward" and distancia == "cosine":
        return ClusteringJerarquico(enlace=enlace, distancia="euclidean"), normalizar_filas(X)
    return ClusteringJerarquico(enlace=enlace, distancia=distancia, p=p), X


def matriz_de_evaluacion(resultado: ResultadoJerarquico, X_modelo, distancia_pedida: str) -> np.ndarray:
    """Matriz de distancias con la que se mide el silhouette.

    Si Ward se ajustó con euclídea sobre filas normalizadas pero se pidió coseno, la partición se
    evalúa igualmente con coseno para que su silhouette sea comparable con el de los demás enlaces.
    """
    if resultado.distancia == distancia_pedida:
        return resultado.distancias_originales
    return ClusteringJerarquico._distancias(X_modelo, distancia_pedida)


def _normas(X) -> np.ndarray:
    if sparse.issparse(X):
        return np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    return np.linalg.norm(np.asarray(X, dtype=float), axis=1)


def _producto(A, B) -> np.ndarray:
    """A @ B.T como array denso, sea cual sea el formato de cada matriz."""
    P = A @ B.T
    return P.toarray() if sparse.issparse(P) else np.asarray(P)


def _distancias_cruzadas(A, B, tipo: str) -> np.ndarray:
    """Distancias entre las filas de A y las de B (coseno o euclídea), con el mismo criterio que fit()."""
    producto = _producto(A, B)
    na, nb = _normas(A), _normas(B)
    if tipo == "cosine":
        denom = np.outer(na, nb)
        similitud = np.divide(producto, denom, out=np.zeros_like(producto), where=denom > 0)
        D = np.clip(1.0 - similitud, 0.0, 2.0)
        D[np.ix_(na == 0, nb == 0)] = 0.0
        return D
    return np.sqrt(np.maximum(na[:, None] ** 2 + nb[None, :] ** 2 - 2 * producto, 0.0))


@dataclass
class ModeloEntrenado:
    """Modelo final que se guarda en modelo.pkl: árbol, datos de entrenamiento y regla para filas nuevas.

    Para cargarlo hace falta que el paquete Clustering_jerarquico sea importable, porque pickle
    guarda la clase por su nombre. Solo deben cargarse archivos .pkl de confianza.
    """
    enlace: str
    distancia: str              # la pedida en la configuración
    distancia_efectiva: str     # la usada para construir el árbol (Ward con coseno => euclidean)
    k: int
    resultado: ResultadoJerarquico
    X_entrenamiento: object     # matriz con la que se ajustó (normalizada si Ward con coseno)
    etiquetas: np.ndarray
    centroides: np.ndarray      # k x atributos, en el espacio de X_entrenamiento
    tamanos: np.ndarray
    metricas: dict
    creado: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def _preparar(self, X_nuevo):
        if sparse.issparse(self.X_entrenamiento):
            X = sparse.csr_matrix(X_nuevo, dtype=np.float64)
            finitos = np.all(np.isfinite(X.data))
        else:
            if sparse.issparse(X_nuevo):
                X_nuevo = X_nuevo.toarray()
            X = np.atleast_2d(np.asarray(X_nuevo, dtype=float))
            finitos = np.all(np.isfinite(X))
        if X.ndim != 2 or X.shape[1] != self.X_entrenamiento.shape[1]:
            raise ValueError(f"Las filas nuevas deben tener {self.X_entrenamiento.shape[1]} atributos "
                             f"(los mismos que el entrenamiento); llegan {X.shape[-1]}.")
        if not finitos:
            raise ValueError("X_nuevo tiene valores NaN o infinitos.")
        if self.distancia != self.distancia_efectiva:     # el modelo se entrenó con filas normalizadas
            X = normalizar_filas(X)
        return X

    def predecir(self, X_nuevo) -> np.ndarray:
        """Asigna un cluster a cada fila nueva.

        El clustering jerárquico no predice de forma nativa, así que cada fila se asigna al cluster
        al que menos le costaría unirse, según el enlace del modelo: single = distancia mínima,
        complete = máxima, average = media y ward = aumento de la suma de cuadrados,
        n_c/(n_c+1)·||x - centroide_c||². Es una aproximación: el árbol no cambia al añadir filas.

        Pre:  X_nuevo tiene los mismos atributos que el entrenamiento y valores finitos.
        Post: vector de enteros en 0..k-1, con las mismas etiquetas que `etiquetas`.
        """
        X = self._preparar(X_nuevo)
        if self.enlace == "ward":
            cuadrados = (_normas(X) ** 2)[:, None] - 2 * _producto(X, self.centroides) \
                + (self.centroides ** 2).sum(axis=1)[None, :]
            coste = np.maximum(cuadrados, 0.0) * (self.tamanos / (self.tamanos + 1.0))[None, :]
        else:
            D = _distancias_cruzadas(X, self.X_entrenamiento, self.distancia_efectiva)
            agregar = {"single": np.min, "complete": np.max, "average": np.mean}[self.enlace]
            coste = np.column_stack([agregar(D[:, self.etiquetas == c], axis=1) for c in range(self.k)])
        return np.argmin(coste, axis=1)

    def resumen(self) -> str:
        m = self.metricas
        fmt = lambda v, d=4: "no calculado" if v is None else f"{v:.{d}f}"
        extra = (f" (calculada como {self.distancia_efectiva} sobre filas normalizadas)"
                 if self.distancia != self.distancia_efectiva else "")
        return "\n".join([
            f"Enlace: {self.enlace} | distancia: {self.distancia}{extra}",
            f"Instancias: {self.resultado.n_instancias} | atributos: {self.X_entrenamiento.shape[1]} | k = {self.k}",
            f"Tamaños de cluster: {self.tamanos.astype(int).tolist()}",
            f"Silhouette: {fmt(m['silhouette'])} | Davies-Bouldin: {fmt(m['davies_bouldin'])} | "
            f"Calinski-Harabasz: {fmt(m['calinski_harabasz'], 2)} | Cofenética: {fmt(m['correlacion_cofenetica'])}",
        ])

    def guardar(self, ruta) -> Path:
        """Guarda el modelo en `ruta` (crea las carpetas que falten) y devuelve la ruta."""
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        with ruta.open("wb") as archivo:
            pickle.dump(self, archivo, protocol=pickle.HIGHEST_PROTOCOL)
        return ruta

    @staticmethod
    def cargar(ruta) -> "ModeloEntrenado":
        """Lee un modelo guardado con `guardar`. Lanza TypeError si el archivo no contiene un ModeloEntrenado."""
        with Path(ruta).open("rb") as archivo:
            modelo = pickle.load(archivo)
        if not isinstance(modelo, ModeloEntrenado):
            raise TypeError(f"{ruta} no contiene un ModeloEntrenado.")
        return modelo


def entrenar_modelo(X, enlace: str, distancia: str, k: int) -> ModeloEntrenado:
    """Ajusta el clustering con la configuración elegida, corta en k clusters y lo empaqueta.

    Pre:  distancia es 'cosine' o 'euclidean', X tiene n filas y 2 <= k < n.
    Post: ModeloEntrenado listo para `guardar` y `predecir`. Lanza ValueError si no se cumple la precondición.
    """
    if distancia not in DISTANCIAS_MODELO:
        raise ValueError(f"Para el modelo final la distancia tiene que ser una de {list(DISTANCIAS_MODELO)}.")

    modelo, X_modelo = preparar_modelo(enlace, distancia, X)
    if sparse.issparse(X_modelo):
        X_modelo = sparse.csr_matrix(X_modelo)
    resultado = modelo.fit(X_modelo)
    n = resultado.n_instancias
    if not 2 <= k < n:
        raise ValueError(f"k tiene que cumplir 2 <= k < {n} (número de instancias).")

    etiquetas = resultado.etiquetas(k)
    tamanos = np.bincount(etiquetas).astype(float)
    miembros = sparse.csr_matrix((np.ones(n), (etiquetas, np.arange(n))), shape=(k, n))
    sumas = miembros @ X_modelo
    centroides = (sumas.toarray() if sparse.issparse(sumas) else np.asarray(sumas)) / tamanos[:, None]

    # Davies-Bouldin y Calinski-Harabasz son euclídeas: con coseno se miden sobre filas normalizadas.
    X_metricas = normalizar_filas(X_modelo) if modelo.distancia == "cosine" else X_modelo
    try:
        db = davies_bouldin(X_metricas, etiquetas)
    except (MemoryError, ValueError):
        db = None
    metricas = {
        "silhouette": silhouette_medio(matriz_de_evaluacion(resultado, X_modelo, distancia), etiquetas),
        "davies_bouldin": db,
        "calinski_harabasz": calinski_harabasz(X_metricas, etiquetas),
        "correlacion_cofenetica": correlacion_cofenetica(resultado),
    }
    return ModeloEntrenado(
        enlace=enlace, distancia=distancia, distancia_efectiva=modelo.distancia, k=int(k),
        resultado=resultado, X_entrenamiento=X_modelo, etiquetas=etiquetas,
        centroides=centroides, tamanos=tamanos, metricas=metricas,
    )
