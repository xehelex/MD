"""Corte del dendrograma por tamaño mínimo de grupo.

Pedir k grupos sobre estos datos devuelve 477/1/1/1, porque el encadenamiento
se gasta el presupuesto despegando gente de una en una. Aquí se piden grupos
de al menos `minimo` miembros y lo que no llega se aparta como atípico.
"""
from __future__ import annotations

import heapq
import itertools

import numpy as np


def _arbol(resultado):
    """Reconstruye hijos, miembros y altura de cada nodo del dendrograma."""
    n = resultado.n_instancias
    hijos, miembros, altura = {}, {i: [i] for i in range(n)}, {i: 0.0 for i in range(n)}

    for idx, (izquierda, derecha, h) in enumerate(resultado.merges):
        nodo = n + idx
        izquierda, derecha = int(izquierda), int(derecha)
        hijos[nodo] = (izquierda, derecha)
        miembros[nodo] = miembros[izquierda] + miembros[derecha]
        altura[nodo] = float(h)

    return hijos, miembros, altura


def cortar_por_tamano(resultado, minimo: int = 10, max_grupos: int | None = None):
    """Devuelve (etiquetas, informe) exigiendo grupos de al menos `minimo`.

    Los apartados van con etiqueta -1. Se baja por las fusiones más altas
    primero. `minimo=1` reproduce el corte clásico.

    Ojo con el umbral: ponerlo alto borra los grupos pequeños, y aquí los
    pequeños son los interesantes. Con 15 se perdía el de ludopatía.
    """
    if minimo < 1:
        raise ValueError("El tamaño mínimo tiene que ser al menos 1.")
    if max_grupos is not None and max_grupos < 1:
        raise ValueError("max_grupos tiene que ser al menos 1.")

    n = resultado.n_instancias
    hijos, miembros, altura = _arbol(resultado)
    raiz = n + len(resultado.merges) - 1

    contador = itertools.count()
    pendientes = [(-altura[raiz], next(contador), raiz)]
    definitivos, apartadas, descartes = [], [], 0

    while pendientes:
        # Los que esperan ya cuentan como grupos.
        if max_grupos is not None and len(definitivos) + len(pendientes) >= max_grupos:
            break

        _, _, nodo = heapq.heappop(pendientes)
        if nodo not in hijos:                      # una hoja no se puede partir
            definitivos.append(nodo)
            continue

        izquierda, derecha = hijos[nodo]
        vale_izq = len(miembros[izquierda]) >= minimo
        vale_der = len(miembros[derecha]) >= minimo

        if vale_izq and vale_der:                  # division util: seguimos por las dos
            heapq.heappush(pendientes, (-altura[izquierda], next(contador), izquierda))
            heapq.heappush(pendientes, (-altura[derecha], next(contador), derecha))
        elif vale_izq:                             # la derecha es residuo
            apartadas.extend(miembros[derecha])
            descartes += 1
            heapq.heappush(pendientes, (-altura[izquierda], next(contador), izquierda))
        elif vale_der:
            apartadas.extend(miembros[izquierda])
            descartes += 1
            heapq.heappush(pendientes, (-altura[derecha], next(contador), derecha))
        else:
            definitivos.append(nodo)               # partirlo lo destruiria

    definitivos.extend(nodo for _, _, nodo in pendientes)

    etiquetas = np.full(n, -1, dtype=int)
    for etiqueta, nodo in enumerate(definitivos):
        for miembro in miembros[nodo]:
            etiquetas[miembro] = etiqueta

    tamanos = sorted((len(miembros[nodo]) for nodo in definitivos), reverse=True)
    informe = {
        "minimo": minimo,
        "grupos": len(definitivos),
        "tamanos": tamanos,
        "atipicas": len(apartadas),
        "atipicas_pct": round(100.0 * len(apartadas) / n, 1),
        "podas": descartes,
    }
    return etiquetas, informe


def k_equivalente(resultado, etiquetas: np.ndarray) -> int:
    """Cuántos grupos pediría un corte clásico para aislar los mismos atípicos.

    Si salen 9 grupos y aquí 46, el clásico habría gastado 37 en gente suelta.
    """
    return int(len(np.unique(etiquetas[etiquetas >= 0])) + (etiquetas == -1).sum())
