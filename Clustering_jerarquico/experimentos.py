"""Experimentación: para cada distancia y enlace se ajusta el árbol una vez, se busca el mejor k con cada
criterio y se guardan las tablas, las etiquetas y las gráficas.

Búsqueda del mejor k ("busqueda" en config.json):
    completa  evalúa todos los k del rango.
    mallado   malla geométrica gruesa y bisección entre los vecinos del mejor punto de la malla.
    binaria   bisección sobre todo el rango; solo es fiable si la curva tiene un único máximo.
"""
import csv
import json
import time
import numpy as np
from tqdm import tqdm
from .algoritmo import ClusteringJerarquico
from .metricas import evaluar_corte, correlacion_cofenetica, adjusted_rand_index, normalizar_filas
from .plots import guardar_dendrograma, guardar_metricas_k, guardar_mds

# Signo para que todos los criterios sean "cuanto más, mejor" (Davies-Bouldin es al revés)
CRITERIOS = {"silhouette": 1, "davies_bouldin": -1, "calinski_harabasz": 1}


def puntuar(valor: float, criterio: str) -> float:
    """Pre: criterio está en CRITERIOS.  Post: puntuación a maximizar (-inf si el valor es NaN)."""
    return CRITERIOS[criterio] * valor if np.isfinite(valor) else -np.inf


# region Búsqueda del mejor k

def buscar_k(puntuacion, k_min: int, k_max: int, metodo: str = "mallado", puntos: int = 12) -> int:
    """Busca el k con mayor puntuación sin tener que probar necesariamente todos.

    Pre:  puntuacion(k) devuelve un número a maximizar, 2 <= k_min <= k_max y metodo es
          'completa', 'mallado' o 'binaria'.
    Post: el mejor k encontrado (en caso de empate, el menor). Con 'completa' es el óptimo exacto.
    """
    if metodo == "completa":
        return max(range(k_min, k_max + 1), key=puntuacion)

    lo, hi = k_min, k_max
    if metodo == "mallado":
        # Malla más densa en los k pequeños (2, 3, 4, 5, 6, 7, 9, 11...) para ver dónde está la zona buena
        malla = sorted(set(np.geomspace(k_min, k_max, puntos).round().astype(int).tolist()))
        m = int(np.argmax([puntuacion(k) for k in malla]))
        lo, hi = malla[max(m - 1, 0)], malla[min(m + 1, len(malla) - 1)]

    # Bisección sobre la pendiente: si k+1 mejora a k, el máximo está a la derecha
    while lo < hi:
        mid = (lo + hi) // 2
        if puntuacion(mid) < puntuacion(mid + 1):
            lo = mid + 1
        else:
            hi = mid
    return lo


def estabilidad(modelo, X, resultado, ks, repeticiones: int, semilla: int) -> dict:
    """Mide si la partición se mantiene al quitar datos.

    Se reajusta el árbol con un 80 % de las filas (sin reemplazo) y se compara su corte con el del
    árbol completo usando el ARI, solo sobre las filas que están en los dos.

    Pre:  resultado es el árbol de modelo ajustado con X, ks son valores de k válidos y repeticiones >= 1.
    Post: {k: ARI medio} (1 = la partición no cambia).
    """
    rng = np.random.default_rng(semilla)
    n = X.shape[0]
    aris = {k: [] for k in ks}
    for _ in range(repeticiones):
        idx = np.sort(rng.choice(n, size=max(int(0.8 * n), max(ks)), replace=False))
        sub = modelo.fit(X[idx])
        for k in ks:
            aris[k].append(adjusted_rand_index(resultado.etiquetas(k)[idx], sub.etiquetas(k)))
    return {k: float(np.mean(v)) for k, v in aris.items()}


def probar_combinacion(X, distancia: str, enlace: str, config: dict, k_min: int, k_max: int):
    """Ajusta el árbol de una combinación y busca el mejor k con cada criterio.

    Pre:  X es la matriz de entrada (ya normalizada si la distancia es coseno), 2 <= k_min <= k_max < n
          y config tiene criterios, busqueda y puntos_malla.
    Post: (árbol, {k: métricas} de los k evaluados, una fila de resumen por criterio).
          Lanza ValueError si la combinación no es válida (por ejemplo ward con manhattan).
    """
    modelo = ClusteringJerarquico(enlace, distancia, config.get("p", 2.0))
    inicio = time.perf_counter()
    resultado = modelo.fit(X)
    segundos = time.perf_counter() - inicio

    # Caché compartida entre criterios: un k ya evaluado no se vuelve a calcular
    evaluados = {}

    def puntuacion(criterio):
        def f(k):
            if k not in evaluados:
                evaluados[k] = evaluar_corte(resultado, X, k)
            return puntuar(evaluados[k][criterio], criterio)
        return f

    criterios = config["criterios"]
    mejores = {c: buscar_k(puntuacion(c), k_min, k_max, config["busqueda"], config["puntos_malla"])
               for c in criterios}
    n_evaluados = len(evaluados)        # se guarda antes de la comprobación, que lo evalúa todo

    # Opcional: comprobar con todos los k si la búsqueda ha acertado
    exhaustivos = ({c: buscar_k(puntuacion(c), k_min, k_max, "completa") for c in criterios}
                   if config.get("comprobar") else {})
    # Opcional: estabilidad con submuestreos (solo para los k elegidos)
    reps = config.get("repeticiones_estabilidad", 0)
    estable = (estabilidad(modelo, X, resultado, set(mejores.values()), reps, config.get("semilla", 42))
               if reps else {})
    cofenetica = correlacion_cofenetica(resultado)

    resumen = [{
        "distancia": distancia, "enlace": enlace, "criterio": c, **evaluados[k],
        "cofenetica": cofenetica, "estabilidad": estable.get(k), "evaluaciones": n_evaluados,
        "tiempo_s": round(segundos, 3), "k_exhaustivo": exhaustivos.get(c),
    } for c, k in mejores.items()]
    return resultado, evaluados, resumen

# endregion


def experimentar(X, config: dict, salida) -> list[dict]:
    """Prueba todas las combinaciones distancia x enlace del config y guarda los resultados.

    Pre:  X es la matriz (n, d) de entrada, config es el contenido de config.json y salida es un Path.
    Post: en salida quedan evaluaciones_k.csv, resumen_mejor_k.csv, resultados.json, las etiquetas
          (.npy) y las gráficas; se imprime la tabla con la sugerencia y se devuelve el resumen.
    """
    n = X.shape[0]
    k_min, k_max = config["k_min"], min(config["k_max"], n - 1)   # no puede haber más clusters que usuarios
    (salida / "plots").mkdir(parents=True, exist_ok=True)
    combinaciones = [(d, e) for d in config["distancias"] for e in config["enlaces"]]
    evaluaciones, resumen = [], []

    barra = tqdm(combinaciones, desc="Experimentando", unit="comb")
    for distancia, enlace in barra:
        barra.set_postfix_str(f"{distancia} / {enlace}")
        # Con coseno normalizamos las filas: las distancias no cambian y DB/CH se miden en ese espacio
        Xc = normalizar_filas(X) if distancia == "cosine" else X
        try:
            resultado, evaluados, filas = probar_combinacion(Xc, distancia, enlace, config, k_min, k_max)
        except ValueError as e:
            tqdm.write(f"Se omite {distancia} / {enlace}: {e}")   # write para no romper la barra
            continue
        por_k = [{"distancia": distancia, "enlace": enlace, **evaluados[k]} for k in sorted(evaluados)]
        evaluaciones += por_k
        resumen += filas

        # Etiquetas y gráficas con el mejor k del primer criterio
        k = filas[0]["k"]
        etiquetas = resultado.etiquetas(k)
        np.save(salida / f"etiquetas_{distancia}_{enlace}_k{k}.npy", etiquetas)
        if config.get("graficas", True):
            carpeta = salida / "plots" / f"{distancia}_{enlace}"
            carpeta.mkdir(exist_ok=True)
            guardar_metricas_k(por_k, {f["criterio"]: f["k"] for f in filas}, f"{distancia} / {enlace}",
                               carpeta / "metricas_por_k.png")
            guardar_dendrograma(resultado, etiquetas, carpeta / "dendrograma.png")
            guardar_mds(resultado.distancias_originales, etiquetas, carpeta / "mds_2d.png")

    if not resumen:
        print("Ninguna combinación válida.")
        return []
    guardar_csv(salida / "evaluaciones_k.csv", evaluaciones)
    guardar_csv(salida / "resumen_mejor_k.csv", resumen)
    parametros = {**{c: v for c, v in config.items() if c != "mejor_modelo"}, "instancias": n, "atributos": X.shape[1]}
    (salida / "resultados.json").write_text(
        json.dumps({"parametros": parametros, "resumen": resumen}, indent=2, ensure_ascii=False), encoding="utf-8")
    imprimir_resumen(resumen, config, n)
    print(f"\nResultados guardados en: {salida.resolve()}")
    return resumen


# region Salida de resultados

def guardar_csv(ruta, filas: list[dict]):
    """Pre: filas no está vacía y todas tienen las mismas claves.  Post: CSV escrito en ruta."""
    with ruta.open("w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)


def imprimir_resumen(resumen: list[dict], config: dict, n: int):
    """Imprime la tabla con el mejor k de cada combinación y criterio, y una sugerencia para config.json.

    Pre:  resumen son las filas que devuelve probar_combinacion y n el número de instancias.
    Post: tabla impresa por pantalla. La sugerencia descarta las filas con * o ~ y solo compara enlaces
          con la misma distancia (el silhouette no es comparable entre distancias distintas).
    """
    criterios = config["criterios"]
    total_k = min(config["k_max"], n - 1) - config["k_min"] + 1
    minimo = max(2, int(np.ceil(0.01 * n)))      # un cluster con menos del 1 % casi siempre es un outlier
    num = lambda v, d=4: "-" if v is None or not np.isfinite(v) else f"{v:.{d}f}"
    aviso = lambda r: ("*" if r["tam_min"] < minimo else "") + ("~" if r["corte_ambiguo"] else "")

    print(f"\n{'Distancia':<10}{'Enlace':<9}{'Criterio':<18}{'k':>3}{'Tamaños':>11}{'Silhouette':>11}"
          f"{'Davies-B.':>10}{'Calinski-H.':>12}{'Cofenética':>11}{'Estab.':>7}{'Evaluac.':>9}")
    for r in sorted(resumen, key=lambda r: (criterios.index(r["criterio"]), r["distancia"], r["enlace"])):
        tamanos = f"{r['tam_min']}/{r['tam_max']}{aviso(r)}"
        linea = (f"{r['distancia']:<10}{r['enlace']:<9}{r['criterio']:<18}{r['k']:>3}{tamanos:>11}"
                 f"{num(r['silhouette']):>11}{num(r['davies_bouldin']):>10}{num(r['calinski_harabasz'], 2):>12}"
                 f"{num(r['cofenetica']):>11}{num(r['estabilidad'], 2):>7}"
                 f"{str(r['evaluaciones']) + '/' + str(total_k):>9}")
        if r["k_exhaustivo"] is not None:
            linea += f"  exhaustivo k={r['k_exhaustivo']}" + ("" if r["k_exhaustivo"] == r["k"] else " (NO coincide)")
        print(linea)
    print("Silhouette y Calinski-H.: mayor es mejor. Davies-B.: menor es mejor.\n"
          f"* algún cluster con menos de {minimo} usuarios (1 %): probables outliers, no grupos.\n"
          "~ corte con alturas empatadas: la partición depende del desempate.")

    # Sugerencia con el primer criterio y la primera distancia que admite el modelo final
    criterio = criterios[0]
    distancia = next((d for d in config["distancias"] if d in ("cosine", "euclidean")), None)
    validas = [r for r in resumen if r["criterio"] == criterio and r["distancia"] == distancia and not aviso(r)]
    if not validas:
        print(f"\nSin sugerencia: ninguna partición válida según '{criterio}' con distancia {distancia}.")
        return
    mejor = max(validas, key=lambda r: puntuar(r[criterio], criterio))
    print(f"\nSugerencia según '{criterio}' ({distancia}): {mejor['enlace']} con k = {mejor['k']} "
          f"(silhouette {num(mejor['silhouette'], 3)}).")
    if mejor["silhouette"] < 0.25:
        print("Silhouette por debajo de 0.25: los grupos son poco claros con esta representación.")
    print(f'Para generar el modelo, en config.json: "mejor_modelo": {{"enlace": "{mejor["enlace"]}", '
          f'"distancia": "{distancia}", "k": {mejor["k"]}, "archivo": "modelo.pkl"}}\n'
          "y ejecuta: python -m Clustering_jerarquico modelo")

# endregion
