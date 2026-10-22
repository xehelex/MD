"""Elegir k mirando los saltos del árbol, en vez de barrer todo el rango.

Donde el árbol pega un salto grande al fusionar es que esos dos grupos estaban
lejos y no convenía juntarlos. Ahí está el corte.
"""
from __future__ import annotations

import numpy as np


def inconsistencia(resultado, profundidad: int = 3) -> np.ndarray:
    """Cuánto destaca cada fusión frente a las de su subárbol.

    El salto en bruto no vale: en la base del árbol son enormes y se lleva el
    premio un k absurdo. Normalizado contra el vecindario, ya no pasa.

    Pre:  resultado viene de fit(); profundidad >= 1.
    Post: array con un valor por fusion, en el mismo orden que merges. Donde
          el vecindario no tiene dispersion devuelve 0, no infinito.
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
    """Candidatos a k, de mejor a peor, mirando la fusión que uniría k con k-1.

    Pre:  2 <= k_min <= k_max < n; cuantos >= 1.
    Post: hasta `cuantos` pares (k, puntuacion) ordenados de mayor a menor,
          todos con k en el rango pedido.

    Da candidatos, no una respuesta: con datos sintéticos de k conocido el bueno
    sale entre los dos primeros, pero no siempre el primero.
    """
    n = resultado.n_instancias
    if not 2 <= k_min <= k_max < n:
        raise ValueError("Tiene que cumplirse 2 <= k_min <= k_max < n.")

    valores = inconsistencia(resultado, profundidad)
    candidatos = [(k, float(valores[n - k])) for k in range(k_min, k_max + 1)]
    candidatos.sort(key=lambda par: par[1], reverse=True)
    return candidatos[:cuantos]


def k_max_recomendado(n: int) -> int:
    """Regla de andar por casa para acotar la busqueda: sqrt(n/2).

    Con 480 usuarios da 15. Evita barrer cientos de k que nadie va a usar.

    Pre:  n >= 1.
    Post: entero >= 2, aunque n sea diminuto.
    """
    return max(2, int(np.sqrt(n / 2)))


def buscar_k_biseccion(criterio, k_min: int = 2, k_max: int = 15) -> tuple[int, float, int]:
    """Mejor k partiendo el rango por la mitad. `criterio` es k -> puntuacion.

    Compara los extremos y se queda con la mitad del que gana: O(log k_max) en
    vez de O(k_max). Devuelve (mejor_k, puntuacion, evaluaciones).

    Pre:  2 <= k_min < k_max y criterio sabe puntuar cualquier k del rango.
    Post: (mejor_k, su puntuacion, cuantos k se evaluaron), con mejor_k en
          [k_min, k_max] y bastantes menos evaluaciones que el barrido.

    Ojo: supone un unico maximo. Con varios picos puede quedarse en el que no
    es, y nuestro silhouette no es unimodal. Para eso esta buscar_k_por_bloques.
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
    """Como la biseccion pero sin jugarsela a que haya un solo pico.

    Prueba unos pocos k por todo el rango, mira cuales destacan sobre sus
    vecinos y afina alrededor de cada uno. Asi no pierde un segundo maximo por
    haber descartado media tabla al principio. Mas caro, pero seguro.

    Pre:  2 <= k_min < k_max y bloques >= 2.
    Post: (mejor_k, su puntuacion, cuantos k se evaluaron). Cuesta mas que la
          biseccion y sigue costando menos que barrer el rango entero.
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
    """El k donde más crece la altura al fusionar.

    Arbol de corte clásico: fusionar k con k-1 es la fusión n-k. La altura de esa
    fusión menos la de la anterior es el salto. Los mayores saltos son candidatos a
    k.

    Pre:  2 <= k_min <= k_max < n-1, que es donde hay dos alturas que restar.
    Post: hasta `cuantos` pares (k, salto) de mayor a menor, con salto >= 0.
    """
    n = resultado.n_instancias
    if not 2 <= k_min <= k_max < n - 1:
        raise ValueError("Tiene que cumplirse 2 <= k_min <= k_max < n-1.")

    alturas = resultado.merges[:, 2]
    candidatos = [(k, float(alturas[n - k - 1] - alturas[n - k - 2]))
                  for k in range(k_min, k_max + 1)]
    candidatos.sort(key=lambda par: par[1], reverse=True)
    return candidatos[:cuantos]
