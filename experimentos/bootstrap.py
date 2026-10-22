"""Bootstrap: cuanto depende el agrupamiento de los usuarios concretos que hay.

Se remuestrea con reemplazo, se rehace el arbol y se compara con la particion
original. Si el ARI se mantiene alto vuelta tras vuelta, el resultado es solido.

Desde consola: python -m experimentos.bootstrap --help
"""
from __future__ import annotations

import argparse
import collections
import sys
from dataclasses import dataclass
from pathlib import Path

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
        """Pre: hay al menos un ARI.  Post: su media, en [-1, 1]."""
        return float(self.aris.mean())

    def intervalo(self, confianza: float = 0.95) -> tuple[float, float]:
        """Entre que valores cae el ARI. Por percentiles, sin suponer normalidad.

        Pre:  0 < confianza < 1.
        Post: (bajo, alto) con bajo <= alto, los dos dentro del rango de aris.
        """
        cola = (1 - confianza) / 2 * 100
        return float(np.percentile(self.aris, cola)), float(np.percentile(self.aris, 100 - cola))

    def ancho(self, confianza: float = 0.95) -> float:
        """Lo que mide el intervalo. Estrecho = da siempre la misma respuesta.

        Mas informativo que la media: una particion degenerada saca ARI alto
        porque es trivial de repetir, pero el ancho es enorme. Un corte solido

        Pre:  0 < confianza < 1.
        Post: valor >= 0. Cuanto mas pequeno, mas repetible es el resultado.
        """
        bajo, alto = self.intervalo(confianza)
        return alto - bajo

    def resumen(self) -> str:
        """Pre: el bootstrap ya corrio.  Post: una linea con media, intervalo y vueltas."""
        bajo, alto = self.intervalo()
        return (f"{self.enlace}/{self.distancia}, k={self.k}: "
                f"ARI {self.media:.3f} [{bajo:.3f}, {alto:.3f}] "
                f"en {len(self.aris)} remuestreos, {self.unicos_medios:.0f} usuarios unicos de media")


def _remuestreo(azar, n: int):
    """Una vuelta: indices con reemplazo, los distintos, y donde mirar cada uno..

    Pre:  n >= 1.
    Post: (indices, distintos, orden) con len(indices) == n. distintos va
          ordenado y sin repetir, y orden dice en que fila de indices mirar
          a cada uno. Quedan fuera un 37% de media.
    """
    indices = azar.integers(0, n, size=n)          # con reemplazo: ~63% de unicos
    distintos = np.unique(indices)
    primera = {i: j for j, i in reversed(list(enumerate(indices)))}
    return indices, distintos, [primera[i] for i in distintos]


def bootstrap(X, enlace: str, distancia: str, k: int, repeticiones: int = 50,
              p: float = 2.0, semilla: int = 42) -> ResultadoBootstrap:
    """Remuestrea con reemplazo y mide cuanto se parece cada particion a la original.

    Mas repeticiones afinan el intervalo: 30 para mirar, 100 para la memoria.

    Pre:  2 <= k < n, repeticiones >= 2 y (enlace, distancia) admitidos.
    Post: ResultadoBootstrap con un ARI por repeticion, todos en [-1, 1].
          Con la misma semilla sale siempre lo mismo.
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
        indices, distintos, orden = _remuestreo(azar, n)
        unicos.append(len(distintos))
        sub = ClusteringJerarquico(enlace=enlace, distancia=distancia, p=p).fit(X[indices])
        aris.append(adjusted_rand_index(referencia[distintos], sub.etiquetas(k)[orden]))

    return ResultadoBootstrap(np.array(aris), k, enlace, distancia, float(np.mean(unicos)))


def bootstrap_por_k(X, resultado, enlace: str, distancia: str, p: float, ks,
                    repeticiones: int = 0, semilla: int = 42) -> dict:
    """Lo mismo que bootstrap() pero para todos los k de una pasada.

    Cada remuestreo construye un arbol y se corta en todos los k.
    Devuelve {k: (media, ancho)}.


    Pre:  resultado es el arbol de X y todos los k de ks cumplen 2 <= k < n.
    Post: {k: (media, ancho)}. Con repeticiones = 0 todo sale nan. Para un k
          suelto da exactamente lo mismo que bootstrap().
    """
    azar = np.random.default_rng(semilla)
    n = X.shape[0]
    acumulado = {k: [] for k in ks}

    for _ in range(repeticiones):
        indices, distintos, orden = _remuestreo(azar, n)
        sub = ClusteringJerarquico(enlace=enlace, distancia=distancia, p=p).fit(X[indices])
        for k in ks:
            acumulado[k].append(
                adjusted_rand_index(resultado.etiquetas(k)[distintos], sub.etiquetas(k)[orden]))

    salida = {}
    for k, valores in acumulado.items():
        if not valores:
            salida[k] = (float("nan"), float("nan"))
            continue
        v = np.array(valores)
        salida[k] = (float(v.mean()),
                     float(np.percentile(v, 97.5) - np.percentile(v, 2.5)))
    return salida


@dataclass
class EstabilidadGrupo:
    grupo: int
    tamano: int
    jaccard: float

    def veredicto(self) -> str:
        """Umbrales de Hennig (2007), los de referencia para esto.

        Pre:  jaccard calculado, no nan.
        Post: "solido" (>=0.85), "aceptable" (>=0.75), "dudoso" (>=0.60) o
              "no se sostiene".
        """
        if self.jaccard >= 0.85:
            return "solido"
        if self.jaccard >= 0.75:
            return "aceptable"
        if self.jaccard >= 0.60:
            return "dudoso"
        return "no se sostiene"


def estabilidad_por_grupo(X, resultado, enlace: str, distancia: str, k: int,
                          repeticiones: int = 50, p: float = 2.0,
                          semilla: int = 42) -> list[EstabilidadGrupo]:
    """Cuanto reaparece cada grupo por separado, en vez de la particion entera.

    Por cada remuestreo se busca el grupo mas parecido a cada original y se mide
    el Jaccard (comunes / totales). 

    Pre:  2 <= k < n y repeticiones >= 2.
    Post: una EstabilidadGrupo por cada uno de los k grupos, con jaccard en
          [0, 1]. Un grupo que no sale en ninguna vuelta da nan.
    """
    n = X.shape[0]
    if repeticiones < 2:
        raise ValueError("Hacen falta al menos 2 repeticiones.")
    if not 2 <= k < n:
        raise ValueError("Tiene que cumplirse 2 <= k < n.")

    referencia = resultado.etiquetas(k)
    azar = np.random.default_rng(semilla)
    acumulado = [[] for _ in range(k)]

    for _ in range(repeticiones):
        indices, distintos, orden = _remuestreo(azar, n)
        sub = ClusteringJerarquico(enlace=enlace, distancia=distancia, p=p).fit(X[indices])
        antes, ahora = referencia[distintos], sub.etiquetas(k)[orden]

        # Tabla de contingencia
        comunes = np.zeros((k, k))
        np.add.at(comunes, (antes, ahora), 1.0)
        totales = comunes.sum(1, keepdims=True) + comunes.sum(0, keepdims=True) - comunes
        jaccard = np.divide(comunes, totales, out=np.zeros_like(comunes), where=totales > 0)

        for grupo in range(k):
            if comunes[grupo].sum() > 0:        # el grupo tiene a alguien esta vuelta
                acumulado[grupo].append(float(jaccard[grupo].max()))

    tamanos = collections.Counter(referencia.tolist())
    return [EstabilidadGrupo(g, tamanos[g], float(np.mean(v)) if v else float("nan"))
            for g, v in enumerate(acumulado)]


ENLACES = ("single", "average", "complete", "ward")


def main(argv=None):
    """Compara la estabilidad de los enlaces por consola.

    Pre:  --entrada tiene la matriz y 2 <= --k < n.
    Post: imprime una tabla con ARI, intervalo, ancho y tamaños por enlace, o
          el Jaccard de cada grupo con --por-grupo.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--entrada", type=Path, required=True,
                        help="Carpeta con X.npz o X.npy y usuarios.json.")
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--repeticiones", type=int, default=50,
                        help="30 para mirar, 100 para la memoria.")
    parser.add_argument("--enlace", choices=ENLACES, action="append",
                        help="Repetible. Sin esto compara los cuatro.")
    parser.add_argument("--distancia", default="cosine")
    parser.add_argument("--p", type=float, default=2.0, help="Solo para minkowski.")
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--por-grupo", action="store_true",
                        help="Estabilidad de cada grupo por separado, en vez del ARI global.")
    args = parser.parse_args(argv)

    from vectorizacion.matrices import cargar_matriz

    from Clustering_jerarquico.algoritmo import ClusteringJerarquico

    try:
        X = cargar_matriz(args.entrada)
    except (OSError, ValueError) as error:
        parser.error(f"No se ha podido cargar {args.entrada}: {error}")
    if not 2 <= args.k < X.shape[0]:
        parser.error(f"k tiene que estar entre 2 y {X.shape[0] - 1}.")

    print(f"\n{args.entrada}  {X.shape[0]} usuarios x {X.shape[1]}  "
          f"k={args.k}  {args.repeticiones} remuestreos")

    if args.por_grupo:
        for enlace in (args.enlace or ENLACES):
            arbol = ClusteringJerarquico(enlace=enlace, distancia=args.distancia,
                                         p=args.p).fit(X)
            grupos = estabilidad_por_grupo(X, arbol, enlace, args.distancia, args.k,
                                           repeticiones=args.repeticiones, p=args.p,
                                           semilla=args.semilla)
            print(f"\n{enlace} / {args.distancia}")
            print(f"{'grupo':>6}{'usuarios':>10}{'jaccard':>9}  veredicto")
            print("-" * 44)
            for g in sorted(grupos, key=lambda g: g.tamano, reverse=True):
                print(f"{g.grupo:>6}{g.tamano:>10}{g.jaccard:>9.3f}  {g.veredicto()}")

        print("\nJaccard = usuarios que el grupo mantiene al rehacerlo con otra muestra.")
        print("Umbrales de Hennig: >=0.85 solido, >=0.75 aceptable, >=0.60 dudoso.")
        print("El tamaño no predice la estabilidad: un grupo chico y apretado aguanta\n"
              "mejor que uno grande y difuso. Dice de que grupos puedes hablar con\n"
              "fuerza al describir los temas, no si el clustering esta bien.\n")

        return 0

    print(f"\n{'enlace':<9}{'ARI':>7}{'intervalo 95%':>18}{'ancho':>8}  tamanos")
    print("-" * 74)

    for enlace in (args.enlace or ENLACES):
        resultado = bootstrap(X, enlace, args.distancia, args.k,
                              repeticiones=args.repeticiones, p=args.p, semilla=args.semilla)
        bajo, alto = resultado.intervalo()
        etiquetas = ClusteringJerarquico(enlace=enlace, distancia=args.distancia,
                                         p=args.p).fit(X).etiquetas(args.k)
        tamanos = sorted(collections.Counter(etiquetas.tolist()).values(), reverse=True)
        print(f"{enlace:<9}{resultado.media:>7.3f}   [{bajo:.3f}, {alto:.3f}]"
              f"{resultado.ancho():>8.3f}  {tamanos[:7]}")

    print(f"\nSe queda fuera un 37% en cada vuelta: {resultado.unicos_medios:.0f} usuarios "
          f"unicos de {X.shape[0]} de media.")
    print("El ancho dice mas que la media: un corte degenerado saca ARI alto porque\n"
          "es trivial de repetir. Mira los tamanos antes de creerte el numero.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
