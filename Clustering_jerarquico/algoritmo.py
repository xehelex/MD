from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools
import numpy as np
from scipy import sparse


@dataclass
class ResultadoJerarquico:
    """Árbol de fusiones (id_izq, id_der, altura) y matriz de distancias original."""
    merges: np.ndarray
    n_instancias: int
    enlace: str
    distancia: str
    distancias_originales: np.ndarray

    def etiquetas(self, k: int) -> np.ndarray:
        """Corta el árbol en exactamente k clusters."""
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
        """Asigna un elemento nuevo al cluster con menor distancia media.

        Es una aproximación, no una predicción nativa del árbol.
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
    """Clustering aglomerativo (single, complete, average, ward).

    Construye la matriz de distancias completa, con coste O(n²) en memoria.
    Manhattan, Chebyshev y Minkowski con p != 2 cuestan O(n²·d).
    """
    ENLACES = {"single", "complete", "average", "ward"}
    DISTANCIAS = {"cosine", "euclidean", "manhattan", "chebyshev", "minkowski"}

    def __init__(self, enlace: str = "average", distancia: str = "cosine", p: float = 2.0):
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
        """Distancias de Minkowski por bloques de filas (~256 MB por bloque)."""
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
        """Matriz de distancias para entrada densa o dispersa."""
        if X.ndim != 2:
            raise ValueError("X tiene que ser una matriz de 2 dimensiones.")
        n = X.shape[0]
        if n < 2:
            raise ValueError("Necesitamos al menos 2 instancias para empezar.")

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
        """Ejecuta el clustering y devuelve el árbol de fusiones."""
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
        """Ajusta el modelo y devuelve las etiquetas para un corte de k clusters."""
        return self.fit(X).etiquetas(k)
