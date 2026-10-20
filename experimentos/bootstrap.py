"""Bootstrap: cuanto depende el agrupamiento de los usuarios concretos que hay.

Se remuestrea con reemplazo, se rehace el arbol y se compara con la particion
original. Si el ARI se mantiene alto vuelta tras vuelta, el resultado es solido.

Ojo con single: los usuarios repetidos quedan a distancia 0 entre si y le
regalan puentes, asi que su estabilidad aqui sale inflada.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from Clustering_jerarquico.algoritmo import ClusteringJerarquico
from Clustering_jerarquico.metricas import adjusted_rand_index


@dataclass
class ResultadoBootstrap:
    aris: np.ndarray
    k: int
    enlace: str
    distancia: str
    unicos_medios: float

    @property
    def media(self) -> float:
        return float(self.aris.mean())

    def intervalo(self, confianza: float = 0.95) -> tuple[float, float]:
        """Entre que valores cae el ARI. Por percentiles, sin suponer normalidad."""
        cola = (1 - confianza) / 2 * 100
        return float(np.percentile(self.aris, cola)), float(np.percentile(self.aris, 100 - cola))

    def resumen(self) -> str:
        bajo, alto = self.intervalo()
        return (f"{self.enlace}/{self.distancia}, k={self.k}: "
                f"ARI {self.media:.3f} [{bajo:.3f}, {alto:.3f}] "
                f"en {len(self.aris)} remuestreos, {self.unicos_medios:.0f} usuarios unicos de media")


def bootstrap(X, enlace: str, distancia: str, k: int, repeticiones: int = 50,
              p: float = 2.0, semilla: int = 42) -> ResultadoBootstrap:
    """Remuestrea con reemplazo y mide cuanto se parece cada particion a la original.

    Mas repeticiones afinan el intervalo: 30 para mirar, 100 para la memoria.
    """
    n = X.shape[0]
    if repeticiones < 2:
        raise ValueError("Hacen falta al menos 2 repeticiones.")
    if not 2 <= k < n:
        raise ValueError("Tiene que cumplirse 2 <= k < n.")

    referencia = ClusteringJerarquico(enlace=enlace, distancia=distancia, p=p).fit(X).etiquetas(k)
    azar = np.random.default_rng(semilla)
    aris, unicos = [], []

    for _ in range(repeticiones):
        indices = azar.integers(0, n, size=n)      # con reemplazo: ~63% de unicos
        distintos = np.unique(indices)
        unicos.append(len(distintos))

        sub = ClusteringJerarquico(enlace=enlace, distancia=distancia, p=p).fit(X[indices])
        etiquetas = sub.etiquetas(k)

        # De los repetidos nos quedamos con su primera aparicion, que si no no
        # se puede comparar uno a uno contra la referencia.
        primera = {i: j for j, i in reversed(list(enumerate(indices)))}
        orden = [primera[i] for i in distintos]
        aris.append(adjusted_rand_index(referencia[distintos], etiquetas[orden]))

    return ResultadoBootstrap(np.array(aris), k, enlace, distancia, float(np.mean(unicos)))
