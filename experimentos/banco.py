"""Banco de pruebas: recorre la rejilla de configuraciones y mide cada partición.

El árbol se construye UNA vez por combinación y se corta en todos los k. Lo
caro es el árbol, no el corte.
"""
from __future__ import annotations

import collections
import time

import numpy as np

from Clustering_jerarquico.algoritmo import ClusteringJerarquico
from Clustering_jerarquico.metricas import (
    adjusted_rand_index, correlacion_cofenetica, silhouette_medio,
)

from .eleccion_k import inconsistencia


def equilibrio(etiquetas: np.ndarray) -> float:
    """Entropía de los tamaños de grupo, normalizada entre 0 y 1.

    1 = todos los grupos iguales, cerca de 0 = uno se lo come todo. Va al lado
    del silhouette porque el silhouette solo no distingue 477/1/1/1 de nada.
    """
    tamanos = np.array(sorted(collections.Counter(np.asarray(etiquetas).tolist()).values()), dtype=float)
    if len(tamanos) < 2:
        return 0.0
    p = tamanos / tamanos.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(tamanos)))


def grupos_utiles(etiquetas: np.ndarray, minimo: int = 10) -> tuple[int, int]:
    """Grupos que superan el mínimo, e instancias que quedan fuera.

    Un 477/1/1/1 son cuatro grupos sobre el papel y uno de verdad.
    """
    cuenta = collections.Counter(np.asarray(etiquetas).tolist())
    utiles = [n for n in cuenta.values() if n >= minimo]
    return len(utiles), int(sum(n for n in cuenta.values() if n < minimo))


def estabilidad_por_k(X, resultado, enlace: str, distancia: str, p: float, ks,
                      repeticiones: int = 5, fraccion: float = 0.8, semilla: int = 42) -> dict:
    """ARI medio entre la partición original y la de un reajuste al 80%.

    El árbol de cada remuestreo se construye una vez y se compara en todos los
    k; si no, esto no entra en la rejilla ni de broma.

    Ojo: una partición degenerada sale estabilísima, porque meter a todos en un
    grupo se reproduce siempre. Mirar junto al equilibrio, nunca sola.
    """
    rng = np.random.default_rng(semilla)
    n = X.shape[0]
    n_sub = max(2, int(np.ceil(fraccion * n)))
    acumulado = {k: [] for k in ks}

    for _ in range(repeticiones):
        indices = np.sort(rng.choice(n, size=min(n_sub, n - 1), replace=False))
        submodelo = ClusteringJerarquico(enlace=enlace, distancia=distancia, p=p)
        subresultado = submodelo.fit(X[indices])
        for k in ks:
            if k < len(indices):
                acumulado[k].append(
                    adjusted_rand_index(resultado.etiquetas(k)[indices], subresultado.etiquetas(k)))

    return {k: float(np.mean(v)) if v else float("nan") for k, v in acumulado.items()}


def evaluar_corte(resultado, etiquetas: np.ndarray, minimo: int) -> dict:
    """Métricas de una partición concreta."""
    cuenta = collections.Counter(etiquetas.tolist())
    mayor = max(cuenta.values())
    utiles, sueltas = grupos_utiles(etiquetas, minimo)
    return {
        "silhouette": silhouette_medio(resultado.distancias_originales, etiquetas),
        "equilibrio": equilibrio(etiquetas),
        "mayor_pct": 100.0 * mayor / len(etiquetas),
        "grupos_utiles": utiles,
        "instancias_sueltas": sueltas,
        "tamanos": " ".join(str(n) for n in sorted(cuenta.values(), reverse=True)[:6]),
    }


def recorrer_rejilla(matrices: dict, combinaciones, ks, minimo: int = 10,
                     repeticiones: int = 0, remuestreos: int = 0) -> list[dict]:
    """Ejecuta la rejilla completa y devuelve una fila por (matriz, enlace, k).

    `matrices` es {nombre: X}, `combinaciones` una lista de (enlace, distancia, p).
    `repeticiones` añade la estabilidad (submuestreo al 80%) y `remuestreos` el
    bootstrap. Las dos multiplican el tiempo por su número.
    """
    # Aqui dentro y no arriba: bootstrap tiene su propio __main__, y si el
    # paquete lo carga al importar, python -m experimentos.bootstrap avisa de
    # que se esta ejecutando dos veces.
    from .bootstrap import bootstrap_por_k

    filas = []
    for nombre, X in matrices.items():
        n = X.shape[0]
        for enlace, distancia, p in combinaciones:
            inicio = time.perf_counter()
            resultado = ClusteringJerarquico(enlace=enlace, distancia=distancia, p=p).fit(X)
            segundos = time.perf_counter() - inicio
            cofenetica = correlacion_cofenetica(resultado)

            validos = [k for k in ks if k < n]
            estable = (estabilidad_por_k(X, resultado, enlace, distancia, p, validos, repeticiones)
                       if repeticiones > 0 else {})
            arranque = (bootstrap_por_k(X, resultado, enlace, distancia, p, validos, remuestreos)
                        if remuestreos > 0 else {})

            # Lo que opina el propio árbol sobre cada corte, sin mirar métricas.
            inc = inconsistencia(resultado)

            for k in validos:
                media, ancho = arranque.get(k, (float("nan"), float("nan")))
                filas.append({
                    "matriz": nombre,
                    "enlace": enlace,
                    "distancia": distancia if distancia != "minkowski" else f"minkowski_p{p:g}",
                    "k": k,
                    **evaluar_corte(resultado, resultado.etiquetas(k), minimo),
                    "estabilidad": estable.get(k, float("nan")),
                    "bootstrap_ari": media,
                    "bootstrap_ancho": ancho,
                    "inconsistencia": float(inc[n - k]),
                    "cofenetica": cofenetica,
                    "segundos_arbol": round(segundos, 3),
                })
    return filas


def correlacion_silhouette_equilibrio(filas: list[dict]) -> float:
    """Pearson entre silhouette y equilibrio sobre toda la rejilla.

    Si sale muy negativa, elegir modelo por silhouette es elegir al revés.
    """
    s = np.array([f["silhouette"] for f in filas], dtype=float)
    e = np.array([f["equilibrio"] for f in filas], dtype=float)
    validos = np.isfinite(s) & np.isfinite(e)
    if validos.sum() < 3 or np.std(s[validos]) == 0 or np.std(e[validos]) == 0:
        return float("nan")
    return float(np.corrcoef(s[validos], e[validos])[0, 1])
