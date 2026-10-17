from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools
import numpy as np
from scipy import sparse


@dataclass
class ResultadoJerarquico:
    """Estructura para guardar el árbol de fusiones y manejar los cortes."""
    merges: np.ndarray          # (n-1, 3): id_izq, id_der, altura
    n_instancias: int
    enlace: str
    distancia: str
    distancias_originales: np.ndarray

    def etiquetas(self, k: int) -> np.ndarray:
        """Corta el árbol para sacar exactamente k clusters."""
        n = self.n_instancias
        if not 1 <= k <= n:
            raise ValueError(f"k tiene que estar entre 1 y {n}.")
        
        grupos = {i: {i} for i in range(n)}
        siguiente_id = n
        
        for izquierda, derecha, _altura in self.merges[:n-k]:
            izquierda, derecha = int(izquierda), int(derecha)
            nuevos = grupos.pop(izquierda) | grupos.pop(derecha)
            grupos[siguiente_id] = nuevos
            siguiente_id += 1
            
        etiquetas = np.empty(n, dtype=int)
        for etiqueta, miembros in enumerate(grupos.values()):
            for miembro in miembros:
                etiquetas[miembro] = etiqueta
        return etiquetas

    def asignar_nuevo(self, distancia_nuevo: np.ndarray, k: int) -> int:
        """Asigna un nuevo elemento al cluster con el que tenga menor distancia media.
        
        Ojo: esto es una aproximación práctica para inferencia rápida, 
        no una predicción nativa del árbol.
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
    """Clustering aglomerativo hecho a mano (single, complete y average linkage).
    
    Nota: Como montamos la matriz de distancias completa, esto chupa O(n²) 
    de memoria. Ideal para datasets de hasta unos miles de elementos.
    """
    ENLACES = {"single", "complete", "average"}
    DISTANCIAS = {"cosine", "euclidean"}

    def __init__(self, enlace: str = "average", distancia: str = "cosine"):
        if enlace not in self.ENLACES:
            raise ValueError(f"Enlace '{enlace}' no válido. Usa: {sorted(self.ENLACES)}")
        if distancia not in self.DISTANCIAS:
            raise ValueError(f"Distancia '{distancia}' no válida. Usa: {sorted(self.DISTANCIAS)}")
            
        self.enlace = enlace
        self.distancia = distancia
        self.resultado_: ResultadoJerarquico | None = None

    @staticmethod
    def _distancias(X, tipo: str) -> np.ndarray:
        """Calcula la matriz de distancias (soporta tanto arrays densos como sparse)."""
        if X.ndim != 2:
            raise ValueError("X tiene que ser una matriz de 2 dimensiones.")
        n = X.shape[0]
        if n < 2:
            raise ValueError("Necesitamos al menos 2 instancias para empezar.")

        if sparse.issparse(X):
            X = X.astype(np.float64, copy=False)
            normas = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
            if tipo == "cosine":
                producto = (X @ X.T).toarray()
                denom = np.outer(normas, normas)
                similitud = np.divide(producto, denom, out=np.zeros_like(producto), where=denom > 0)
                D = np.clip(1.0 - similitud, 0.0, 2.0)
                
                # Vectores a cero: si ambos son cero, distancia 0; si uno lo es, distancia 1.
                ceros = normas == 0
                D[np.ix_(ceros, ceros)] = 0.0
            else:
                producto = (X @ X.T).toarray()
                norma2 = normas ** 2
                d2 = np.maximum(norma2[:, None] + norma2[None, :] - 2 * producto, 0.0)
                D = np.sqrt(d2)
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
        """Ejecuta el clustering y devuelve las fusiones del árbol."""
        D0 = self._distancias(X, self.distancia)
        n = D0.shape[0]
        
        # Diccionario para llevar las distancias activas entre clusters
        dist = {}
        for i in range(n):
            for j in range(i + 1, n):
                dist[(i, j)] = float(D0[i, j])
                
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
                raise RuntimeError("Vaya, no se ha podido encontrar una fusión válida.")
                
            merges.append((i, j, d))
            nuevos = next_id
            next_id += 1
            otros = list(active - {i, j})
            size_i, size_j = sizes[i], sizes[j]

            # Actualizamos distancias según el criterio de enlace elegido
            for z in otros:
                diz = dist[(min(i, z), max(i, z))]
                djz = dist[(min(j, z), max(j, z))]
                
                if self.enlace == "single":
                    dnz = min(diz, djz)
                elif self.enlace == "complete":
                    dnz = max(diz, djz)
                else:  # average
                    dnz = (size_i * diz + size_j * djz) / (size_i + size_j)
                    
                key = (min(nuevos, z), max(nuevos, z))
                dist[key] = float(dnz)
                heapq.heappush(heap, (float(dnz), next(counter), key[0], key[1]))

            active.remove(i)
            active.remove(j)
            active.add(nuevos)
            sizes[nuevos] = size_i + size_j
            
            # Limpiamos las distancias viejas de los nodos que acabamos de fusionar
            for key in list(dist):
                if i in key or j in key:
                    del dist[key]

        self.resultado_ = ResultadoJerarquico(
            merges=np.asarray(merges, dtype=float),
            n_instancias=n,
            enlace=self.enlace,
            distancia=self.distancia,
            distancias_originales=D0,
        )
        return self.resultado_

    def fit_predict(self, X, k: int) -> np.ndarray:
        """Ajusta el modelo y devuelve directamente las etiquetas para un corte de k clusters."""
        return self.fit(X).etiquetas(k)