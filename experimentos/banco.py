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
from Clustering_jerarquico.metricas import (
    adjusted_rand_index, correlacion_cofenetica, silhouette_medio,
)


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


def estabilidad_por_k(X, resultado, enlace: str, distancia: str, p: float, ks,
                      repeticiones: int = 5, fraccion: float = 0.8, semilla: int = 42) -> dict:
    """ARI medio entre la partición de referencia y la de un reajuste parcial.

    En cada repetición se retira al azar el 20% de las filas, se reconstruye el
    árbol con el resto y se compara contra la partición original restringida a
    esas mismas filas. Si el modelo es estable, quitar una quinta parte de los
    usuarios no deberia reorganizar los grupos.

    El árbol de cada remuestreo se construye una sola vez y se compara en todos
    los k, que es lo que hace viable meter esto en la rejilla.

    Ojo al leerlo: una partición degenerada es trivialmente estable, porque
    meter a todo el mundo en un grupo se reproduce siempre. Hay que mirarlo
    junto al equilibrio, nunca solo.
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
                     repeticiones: int = 0) -> list[dict]:
    """Ejecuta la rejilla completa y devuelve una fila por (matriz, enlace, k).

    `matrices` es {nombre: X}. `combinaciones` es una lista de (enlace, distancia, p).
    Cada árbol se construye una sola vez y se reutiliza para todos los k. Con
    `repeticiones` > 0 se añade la columna de estabilidad, que multiplica el
    tiempo por ese número porque obliga a reconstruir árboles.
    """
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

            for k in validos:
                filas.append({
                    "matriz": nombre,
                    "enlace": enlace,
                    "distancia": distancia if distancia != "minkowski" else f"minkowski_p{p:g}",
                    "k": k,
                    **evaluar_corte(resultado, resultado.etiquetas(k), minimo),
                    "estabilidad": estable.get(k, float("nan")),
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
