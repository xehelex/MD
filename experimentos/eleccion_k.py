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


def k_max_recomendado(n: int) -> int:
    """Regla habitual para acotar la busqueda: sqrt(n/2).

    Con 480 usuarios da 15. No tiene mas fundamento que la experiencia, pero
    evita barrer cientos de k que nadie va a usar.
    """
    return max(2, int(np.sqrt(n / 2)))


def buscar_k_biseccion(criterio, k_min: int = 2, k_max: int = 15) -> tuple[int, float, int]:
    """Busca el mejor k partiendo el rango por la mitad en cada paso.

    `criterio` es una funcion k -> puntuacion, a maximizar. Devuelve
    (mejor_k, puntuacion, evaluaciones).

    En cada vuelta compara los dos extremos y se queda con la mitad del que va
    ganando. Pasa de O(k_max) evaluaciones a O(log k_max).

    OJO: esto da por supuesto que la curva tiene un unico maximo. Si tiene
    varios picos puede quedarse en el que no es. Sobre nuestros datos el
    silhouette NO es unimodal, asi que conviene contrastarlo con el barrido
    completo antes de fiarse.
    """
    if not 2 <= k_min < k_max:
        raise ValueError("Tiene que cumplirse 2 <= k_min < k_max.")

    cache, evaluaciones = {}, 0

    def puntuar(k):
        nonlocal evaluaciones
        if k not in cache:
            cache[k] = criterio(k)
            evaluaciones += 1
        return cache[k]

    bajo, alto = k_min, k_max
    while alto - bajo > 1:
        medio = (bajo + alto) // 2
        if puntuar(bajo) >= puntuar(alto):
            alto = medio
        else:
            bajo = medio

    mejor = max((bajo, alto), key=puntuar)
    return mejor, puntuar(mejor), evaluaciones


def buscar_k_por_bloques(criterio, k_min: int = 2, k_max: int = 15,
                         bloques: int = 4) -> tuple[int, float, int]:
    """Como la biseccion pero sin jugarsela a que la curva tenga un solo pico.

    Primero prueba unos pocos k repartidos por todo el rango. Mira cuales de
    esos destacan sobre sus vecinos, y afina alrededor de cada uno. Asi un
    segundo maximo no se pierde por haber descartado media tabla al principio.

    Devuelve (mejor_k, puntuacion, evaluaciones). Sale mas caro que la
    biseccion, pero sigue muy por debajo del barrido completo.
    """
    if not 2 <= k_min < k_max:
        raise ValueError("Tiene que cumplirse 2 <= k_min < k_max.")
    if bloques < 2:
        raise ValueError("Hacen falta al menos 2 bloques.")

    cache = {}

    def puntuar(k):
        if k not in cache:
            cache[k] = criterio(k)
        return cache[k]

    # 1) muestreo grueso, los bordes incluidos
    marcas = sorted({int(round(v)) for v in np.linspace(k_min, k_max, bloques + 1)})

    # 2) los que no bajan respecto a sus vecinos son zona sospechosa
    candidatos = []
    for i, k in enumerate(marcas):
        izquierda = puntuar(marcas[i - 1]) if i > 0 else -np.inf
        derecha = puntuar(marcas[i + 1]) if i < len(marcas) - 1 else -np.inf
        if puntuar(k) >= izquierda and puntuar(k) >= derecha:
            candidatos.append(i)

    # 3) alrededor de cada sospechoso, mirar uno a uno: son tramos cortos
    for i in candidatos:
        desde = marcas[i - 1] if i > 0 else k_min
        hasta = marcas[i + 1] if i < len(marcas) - 1 else k_max
        for k in range(desde, hasta + 1):
            puntuar(k)

    mejor = max(cache, key=cache.get)
    return mejor, cache[mejor], len(cache)


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
