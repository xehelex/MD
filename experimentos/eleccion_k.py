"""Elegir k leyendo el árbol, sin barrer el rango entero.

Barrer k y quedarse con el mejor silhouette aquí no funciona: sale k=2 con el
99,8% de la gente en un grupo. Y acelerar esa búsqueda tampoco arregla nada,
porque el problema es el criterio, no el recorrido.

El dendrograma ya lleva la respuesta dentro: donde el árbol da un salto grande
al fusionar, es que esos dos grupos estaban lejos y no convenía juntarlos.
"""
from __future__ import annotations

import numpy as np


def inconsistencia(resultado, profundidad: int = 3) -> np.ndarray:
    """Cuánto destaca cada fusión frente a las de su propio subárbol.

    El salto en bruto no sirve: abajo del todo, donde se juntan usuarios casi
    idénticos, los saltos relativos son enormes y se lleva el premio k=475.
    Normalizando contra el vecindario de cada fusión, eso deja de pasar.
    """
    n = resultado.n_instancias
    merges = resultado.merges

    hijos = {n + i: (int(a), int(b)) for i, (a, b, _h) in enumerate(merges)}
    altura = {n + i: float(h) for i, (_a, _b, h) in enumerate(merges)}

    valores = np.zeros(len(merges))
    for i in range(len(merges)):
        pila, vecindario = [(n + i, 0)], []
        while pila:
            nodo, prof = pila.pop()
            if nodo in altura and prof <= profundidad:
                vecindario.append(altura[nodo])
                if nodo in hijos:
                    pila.extend((h, prof + 1) for h in hijos[nodo])
        desviacion = np.std(vecindario)
        valores[i] = (altura[n + i] - np.mean(vecindario)) / desviacion if desviacion > 1e-12 else 0.0

    return valores


def sugerir_k(resultado, k_min: int = 2, k_max: int = 30,
              profundidad: int = 3, cuantos: int = 3) -> list[tuple[int, float]]:
    """Propone candidatos a k, de mejor a peor, con su puntuación.

    Mira la fusión que uniría k grupos en k-1: si destaca mucho sobre su
    vecindario, es que no convenía hacerla y ahí está el corte.

    Da candidatos, no una respuesta. Probado con datos sintéticos de k conocido,
    el k verdadero sale entre los dos primeros casi siempre, pero no siempre el
    primero. Sirve para acotar el rango, no para cerrar la decisión.

    Con single y average devuelve cualquier cosa, pero es que ahí no hay
    estructura que encontrar.
    """
    n = resultado.n_instancias
    if not 2 <= k_min <= k_max < n:
        raise ValueError("Tiene que cumplirse 2 <= k_min <= k_max < n.")

    valores = inconsistencia(resultado, profundidad)
    candidatos = [(k, float(valores[n - k])) for k in range(k_min, k_max + 1)]
    candidatos.sort(key=lambda par: par[1], reverse=True)
    return candidatos[:cuantos]


def salto_mayor(resultado, k_min: int = 2, k_max: int = 30,
                cuantos: int = 3) -> list[tuple[int, float]]:
    """Variante simple: el k donde más crece la altura al fusionar.

    Más directa de explicar que la inconsistencia, pero se deja engañar por la
    base del árbol, así que hay que acotarla a la zona alta sí o sí.
    """
    n = resultado.n_instancias
    if not 2 <= k_min <= k_max < n - 1:
        raise ValueError("Tiene que cumplirse 2 <= k_min <= k_max < n-1.")

    alturas = resultado.merges[:, 2]
    candidatos = [(k, float(alturas[n - k - 1] - alturas[n - k - 2]))
                  for k in range(k_min, k_max + 1)]
    candidatos.sort(key=lambda par: par[1], reverse=True)
    return candidatos[:cuantos]
