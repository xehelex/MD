"""Corte del dendrograma por tamaño mínimo de grupo.

Pedir k grupos aquí devuelve 477/1/1/1: el encadenamiento se gasta el
presupuesto despegando gente de una en una. Esto pide grupos de al menos
`minimo` miembros y aparta el resto como atípico.
"""
from __future__ import annotations

import heapq
import itertools

import numpy as np


def _arbol(resultado):
    """Reconstruye hijos, miembros y altura de cada nodo del dendrograma.

    Pre:  resultado viene de fit(); sus merges estan en orden de fusion.
    Post: (hijos, miembros, altura) indexados por nodo. Las hojas 0..n-1 no
          aparecen en hijos y tienen altura 0.
    """
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

    Los apartados se marcan con -1. Si se da max_grupos, no se crean mas de esos.

    Pre:  minimo >= 1 y max_grupos >= 1 si se da.
    Post: (etiquetas, informe). etiquetas tiene n enteros: 0..g-1 para los
          grupos y -1 para los apartados. Todo grupo con etiqueta >= 0 tiene
          al menos `minimo` miembros, y g <= max_grupos si se pidio.
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
    """Cuántos grupos pediría el corte clásico para aislar los mismos atípicos.

    Pre:  etiquetas sale de cortar_por_tamano, con -1 en los apartados.
    Post: numero de grupos mas numero de apartados, que es el k que haria
          falta pedirle al corte clasico para separarlos uno a uno.
    """
    return int(len(np.unique(etiquetas[etiquetas >= 0])) + (etiquetas == -1).sum())
