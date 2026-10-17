"""Banco de pruebas: recorre la rejilla de configuraciones y mide cada partición.

El árbol de cada combinación de enlace y distancia se construye UNA vez y luego
se corta en todos los k pedidos, que es lo que hace barato el barrido: lo caro es
el árbol, no el corte.
"""
from __future__ import annotations

import collections
import time

import numpy as np

from Clustering_jerarquico.algoritmo import ClusteringJerarquico
from Clustering_jerarquico.metricas import correlacion_cofenetica, silhouette_medio


def equilibrio(etiquetas: np.ndarray) -> float:
    """Entropía de los tamaños de grupo, normalizada entre 0 y 1.

    Vale 1 cuando todos los grupos tienen el mismo tamaño y tiende a 0 cuando uno
    se lo come todo. Hace falta porque el silhouette por sí solo no distingue una
    partición repartida de otra con 477 instancias en un grupo y tres sueltas:
    de hecho puntúa mejor la segunda.
    """
    tamanos = np.array(sorted(collections.Counter(np.asarray(etiquetas).tolist()).values()), dtype=float)
    if len(tamanos) < 2:
        return 0.0
    p = tamanos / tamanos.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(tamanos)))


def grupos_utiles(etiquetas: np.ndarray, minimo: int = 10) -> tuple[int, int]:
    """Cuántos grupos superan el tamaño mínimo y cuántas instancias quedan fuera.

    Un corte en k=4 que devuelve 477/1/1/1 tiene formalmente cuatro grupos pero
    solo uno aprovechable; esto lo deja a la vista en la tabla.
    """
    cuenta = collections.Counter(np.asarray(etiquetas).tolist())
    utiles = [n for n in cuenta.values() if n >= minimo]
    return len(utiles), int(sum(n for n in cuenta.values() if n < minimo))


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


def recorrer_rejilla(matrices: dict, combinaciones, ks, minimo: int = 10) -> list[dict]:
    """Ejecuta la rejilla completa y devuelve una fila por (matriz, enlace, k).

    `matrices` es {nombre: X}. `combinaciones` es una lista de (enlace, distancia, p).
    Cada árbol se construye una sola vez y se reutiliza para todos los k.
    """
    filas = []
    for nombre, X in matrices.items():
        n = X.shape[0]
        for enlace, distancia, p in combinaciones:
            inicio = time.perf_counter()
            resultado = ClusteringJerarquico(enlace=enlace, distancia=distancia, p=p).fit(X)
            segundos = time.perf_counter() - inicio
            cofenetica = correlacion_cofenetica(resultado)

            for k in ks:
                if k >= n:
                    continue
                filas.append({
                    "matriz": nombre,
                    "enlace": enlace,
                    "distancia": distancia if distancia != "minkowski" else f"minkowski_p{p:g}",
                    "k": k,
                    **evaluar_corte(resultado, resultado.etiquetas(k), minimo),
                    "cofenetica": cofenetica,
                    "segundos_arbol": round(segundos, 3),
                })
    return filas


def correlacion_silhouette_equilibrio(filas: list[dict]) -> float:
    """Pearson entre silhouette y equilibrio sobre toda la rejilla.

    Si sale muy negativa, el silhouette está premiando justo las particiones que
    no sirven, y elegir modelo por esa métrica es elegir al revés.
    """
    s = np.array([f["silhouette"] for f in filas], dtype=float)
    e = np.array([f["equilibrio"] for f in filas], dtype=float)
    validos = np.isfinite(s) & np.isfinite(e)
    if validos.sum() < 3 or np.std(s[validos]) == 0 or np.std(e[validos]) == 0:
        return float("nan")
    return float(np.corrcoef(s[validos], e[validos])[0, 1])
