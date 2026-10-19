"""Experimentación: clustering por enlace, búsqueda del mejor k y tablas comparativas.

Métodos de búsqueda del mejor k (--busqueda):
    mallado  malla geométrica gruesa para localizar la zona buena y bisección dentro de ella (por defecto).
    binaria  bisección pura sobre todo el rango; solo es fiable si la curva tiene un único máximo.
"""
from __future__ import annotations
import csv
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from .algoritmo import DISTANCIAS_MODELO, ClusteringJerarquico, matriz_de_evaluacion, preparar_modelo
from .metricas import (silhouette_medio, correlacion_cofenetica, davies_bouldin,
                       calinski_harabasz, adjusted_rand_index, normalizar_filas)
from .plots import guardar_dendrograma, guardar_silhouette, guardar_proyeccion_mds, guardar_busqueda_k

# Signo con el que cada métrica se convierte en una puntuación a maximizar.
CRITERIOS = {"silhouette": 1.0, "davies_bouldin": -1.0, "calinski_harabasz": 1.0}

# Un cluster con menos del 1 % de las instancias casi siempre es un outlier y no un grupo.
FRACCION_MIN_CLUSTER = 0.01

# Regla orientativa habitual: por debajo de 0.25 el silhouette indica que apenas hay estructura de grupos.
SILHOUETTE_DEBIL = 0.25

COLUMNAS_EVALUACIONES = [
    "distancia", "distancia_efectiva", "enlace", "k", "silhouette", "davies_bouldin",
    "calinski_harabasz", "tam_min", "tam_max", "corte_ambiguo",
    "visitado_silhouette", "visitado_davies_bouldin", "visitado_calinski_harabasz",
]
COLUMNAS_RESUMEN = [
    "distancia", "distancia_efectiva", "enlace", "busqueda", "criterio", "k_mejor",
    "silhouette", "davies_bouldin", "calinski_harabasz", "tam_min", "tam_max", "corte_ambiguo",
    "evaluaciones_busqueda", "evaluaciones_barrido",
    "tiempo_fit_s", "tiempo_busqueda_s", "correlacion_cofenetica",
    "k_exhaustivo", "valor_exhaustivo", "busqueda_acierta", "k_visitados",
]


class Progreso:
    """Línea de progreso con el paso actual, una barra y la tarea en curso.

    En una terminal interactiva la línea se reescribe en el sitio. Si la salida se redirige a un archivo
    o la consola no admite el retorno de carro, se imprime una línea por paso. Con `visible=False` no
    muestra nada, pero `escribir` sigue imprimiendo los resultados.
    """
    ANCHO_BARRA = 20

    def __init__(self, total: int = 0, visible: bool = True):
        self.total = total
        self.paso_actual = 0
        self.texto = ""
        self.visible = visible
        self.en_vivo = visible and sys.stdout.isatty()

    def _ancho(self) -> int:
        return shutil.get_terminal_size((100, 20)).columns - 1

    def _linea(self) -> str:
        hechos = max(self.paso_actual - 1, 0)
        lleno = round(self.ANCHO_BARRA * hechos / self.total) if self.total else 0
        barra = "#" * lleno + "-" * (self.ANCHO_BARRA - lleno)
        return f"[{self.paso_actual}/{self.total}] [{barra}] {self.texto}"

    def _pintar(self):
        if self.en_vivo and self.paso_actual:
            print("\r" + self._linea()[:self._ancho()].ljust(self._ancho()), end="", flush=True)

    def _borrar(self):
        if self.en_vivo:
            print("\r" + " " * self._ancho() + "\r", end="", flush=True)

    def paso(self, texto: str):
        """Empieza un paso nuevo."""
        self.paso_actual += 1
        self.texto = texto
        if self.visible and not self.en_vivo:
            print(self._linea(), flush=True)
        self._pintar()

    def detalle(self, texto: str):
        """Cambia el texto del paso en curso (solo se ve en una terminal interactiva)."""
        self.texto = texto
        self._pintar()

    def escribir(self, texto: str = ""):
        """Imprime un mensaje o una tabla sin estropear la línea de progreso."""
        self._borrar()
        print(texto, flush=True)
        self._pintar()

    def terminar(self):
        """Quita la línea de progreso."""
        self._borrar()
        self.paso_actual = 0


# Búsqueda del mejor k

def _biseccion(f, lo: int, hi: int) -> int:
    """Bisección sobre la pendiente de la curva f (a maximizar) en [lo, hi].

    Compara k = mid con k = mid + 1: si mejora, el óptimo está a la derecha; si no, está en mid o a su
    izquierda. Cada paso descarta la mitad del intervalo. En caso de empate gana el k menor.
    """
    while lo < hi:
        mid = (lo + hi) // 2
        if f(mid) < f(mid + 1):
            lo = mid + 1
        else:
            hi = mid
    f(lo)
    return lo


def malla_geometrica(k_min: int, k_max: int, puntos: int) -> list[int]:
    """Valores de k espaciados geométricamente, más densos donde k es pequeño."""
    ks = np.unique(np.round(np.geomspace(k_min, k_max, puntos)).astype(int))
    return [int(k) for k in ks]


def buscar_mejor_k(puntuacion, k_min: int, k_max: int, metodo: str = "mallado",
                   puntos: int = 12) -> tuple[int, list[int]]:
    """Devuelve el mejor k y los k evaluados, en orden de visita.

    'binaria': bisección pura sobre [k_min, k_max], con unas 2·log2(k_max - k_min) evaluaciones. Asume un único
    máximo; con varios puede quedarse en uno que no es el global.
    'mallado': evalúa una malla geométrica de `puntos` valores, toma el mejor y hace la bisección solo entre
    sus dos vecinos de la malla.

    Pre:  `puntuacion(k)` devuelve un número a maximizar y 2 <= k_min <= k_max.
    """
    visitados: list[int] = []

    def f(k: int) -> float:
        if k not in visitados:
            visitados.append(k)
        return puntuacion(k)

    lo, hi = k_min, k_max
    if metodo == "mallado":
        malla = malla_geometrica(k_min, k_max, puntos)
        mejor = int(np.argmax([f(k) for k in malla]))
        lo, hi = malla[max(mejor - 1, 0)], malla[min(mejor + 1, len(malla) - 1)]

    return _biseccion(f, lo, hi), visitados


def _puntuacion(evaluar, criterio: str):
    signo = CRITERIOS[criterio]

    def f(k: int) -> float:
        valor = evaluar(k)[criterio]
        return signo * valor if np.isfinite(valor) else -np.inf

    return f


def tam_minimo_valido(n: int) -> int:
    """Tamaño mínimo que debe tener cada cluster para que la partición se considere razonable."""
    return max(2, int(np.ceil(FRACCION_MIN_CLUSTER * n)))


def corte_ambiguo(resultado, k: int) -> bool:
    """True si las dos fusiones a ambos lados del corte tienen la misma altura.

    Entonces el árbol no decide qué partición de k clusters es la correcta: depende del orden de
    desempate. Ocurre, por ejemplo, con complete cuando varias fusiones llegan a la distancia máxima.
    """
    alturas, n = resultado.merges[:, 2], resultado.n_instancias
    return bool(2 <= k < n and np.isclose(alturas[n - k - 1], alturas[n - k], rtol=0, atol=1e-12))


def probar_combinacion(modelo, X, k_min: int, k_max: int, criterios: list[str], comprobar: bool, metodo: str,
                       puntos: int, distancia_solicitada: str | None = None,
                       progreso: Progreso | None = None, etiqueta: str = ""):
    """Ajusta el modelo una vez y busca el mejor k para cada criterio.

    Las evaluaciones se comparten entre criterios: un k ya evaluado no se recalcula.
    Devuelve (filas con todas las evaluaciones, resumen con el mejor k de cada criterio).
    """
    progreso = progreso or Progreso(visible=False)
    enlace = modelo.enlace
    efectiva = modelo.distancia
    distancia = distancia_solicitada or efectiva

    progreso.detalle(f"{etiqueta} · ajustando el árbol")
    inicio = time.perf_counter()
    resultado = modelo.fit(X)
    t_fit = time.perf_counter() - inicio

    D = matriz_de_evaluacion(resultado, X, distancia)
    X_db = normalizar_filas(X) if efectiva == "cosine" else X
    cache: dict[int, dict] = {}

    def evaluar(k: int) -> dict:
        if k not in cache:
            progreso.detalle(f"{etiqueta} · evaluando k = {k}")
            etiquetas = resultado.etiquetas(k)
            tam = np.bincount(etiquetas)
            cache[k] = {
                "k": k,
                "silhouette": silhouette_medio(D, etiquetas),
                "davies_bouldin": davies_bouldin(X_db, etiquetas),
                "calinski_harabasz": calinski_harabasz(X_db, etiquetas),
                "tam_min": int(tam.min()),
                "tam_max": int(tam.max()),
                "corte_ambiguo": corte_ambiguo(resultado, k),
            }
        return cache[k]

    inicio = time.perf_counter()
    busquedas = {}
    for criterio in criterios:
        k_mejor, visitados = buscar_mejor_k(_puntuacion(evaluar, criterio), k_min, k_max, metodo, puntos)
        busquedas[criterio] = {"k_mejor": k_mejor, "visitados": visitados}
    t_busqueda = time.perf_counter() - inicio

    exhaustivos = {}
    if comprobar:
        for k in range(k_min, k_max + 1):
            evaluar(k)
        for criterio in criterios:
            f = _puntuacion(evaluar, criterio)
            exhaustivos[criterio] = max(range(k_min, k_max + 1), key=f)

    cofenetica = correlacion_cofenetica(resultado)
    total_barrido = k_max - k_min + 1

    filas = []
    for k in sorted(cache):
        fila = {"distancia": distancia, "distancia_efectiva": efectiva, "enlace": enlace, **cache[k]}
        for criterio in CRITERIOS:
            fila[f"visitado_{criterio}"] = criterio in busquedas and k in busquedas[criterio]["visitados"]
        filas.append(fila)

    resumen = []
    for criterio, datos in busquedas.items():
        mejor = cache[datos["k_mejor"]]
        fila = {
            "distancia": distancia,
            "distancia_efectiva": efectiva,
            "enlace": enlace,
            "busqueda": metodo,
            "criterio": criterio,
            "k_mejor": datos["k_mejor"],
            "silhouette": mejor["silhouette"],
            "davies_bouldin": mejor["davies_bouldin"],
            "calinski_harabasz": mejor["calinski_harabasz"],
            "tam_min": mejor["tam_min"],
            "tam_max": mejor["tam_max"],
            "corte_ambiguo": mejor["corte_ambiguo"],
            "evaluaciones_busqueda": len(datos["visitados"]),
            "evaluaciones_barrido": total_barrido,
            "tiempo_fit_s": round(t_fit, 4),
            "tiempo_busqueda_s": round(t_busqueda, 4),
            "correlacion_cofenetica": cofenetica,
            "k_visitados": datos["visitados"],
        }
        if comprobar:
            k_exh = exhaustivos[criterio]
            f = _puntuacion(evaluar, criterio)
            fila["k_exhaustivo"] = k_exh
            fila["valor_exhaustivo"] = cache[k_exh][criterio]
            fila["busqueda_acierta"] = bool(f(datos["k_mejor"]) >= f(k_exh) - 1e-12)
        resumen.append(fila)

    return filas, resumen


def guardar_resultados(carpeta: Path, parametros: dict, filas: list[dict], resumen: list[dict]):
    """Escribe evaluaciones_k.csv, resumen_mejor_k.csv y resultados.json en `carpeta`."""
    with (carpeta / "evaluaciones_k.csv").open("w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=COLUMNAS_EVALUACIONES, extrasaction="ignore")
        escritor.writeheader()
        escritor.writerows(filas)

    with (carpeta / "resumen_mejor_k.csv").open("w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=COLUMNAS_RESUMEN, extrasaction="ignore")
        escritor.writeheader()
        for fila in resumen:
            escritor.writerow({**fila, "k_visitados": ";".join(map(str, fila["k_visitados"]))})

    contenido = {"parametros": parametros, "resumen": resumen}
    (carpeta / "resultados.json").write_text(
        json.dumps(contenido, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )


# Presentación de resultados

def _num(valor, decimales: int = 4) -> str:
    if valor is None or not np.isfinite(valor):
        return "-"
    return f"{valor:.{decimales}f}"


def formatear_tabla(cabeceras: list[str], filas: list[list[str]], texto: int = 2) -> str:
    """Tabla de texto: las `texto` primeras columnas a la izquierda y el resto (números) a la derecha."""
    anchos = [max(len(str(c)) for c in col) for col in zip(cabeceras, *filas)]

    def linea(celdas):
        return "  ".join(str(c).ljust(a) if i < texto else str(c).rjust(a)
                         for i, (c, a) in enumerate(zip(celdas, anchos)))

    return "\n".join([linea(cabeceras), "  ".join("-" * a for a in anchos)] + [linea(f) for f in filas])


def _aviso(tam_min: int, minimo: int, ambiguo: bool) -> str:
    return (" *" if tam_min < minimo else "") + (" ~" if ambiguo else "")


def _leyenda(minimo: int, hay_pequenos: bool, hay_ambiguos: bool, hay_ward_coseno: bool = False) -> list[str]:
    lineas = []
    if hay_pequenos:
        lineas.append(f"* algún cluster con menos de {minimo} usuarios (1 %): probables outliers, no grupos.")
    if hay_ambiguos:
        lineas.append("~ corte con alturas empatadas: la partición depende del desempate.")
    if hay_ward_coseno:
        lineas.append("Ward con coseno se calcula con distancia euclídea sobre filas normalizadas (equivalente).")
    if hay_pequenos or hay_ambiguos:
        lineas.append("Las filas con * o ~ no se sugieren.")
    return lineas


def sugerir_mejor(resumen: list[dict], criterio: str, distancia: str, minimo: int) -> dict | None:
    """Fila del resumen con la mejor puntuación según `criterio`, o None si ninguna es válida.

    Solo compara enlaces con la misma `distancia`. Descarta las particiones con algún cluster de menos de
    `minimo` instancias y los cortes ambiguos por empate de alturas.
    """
    candidatas = [r for r in resumen if r["criterio"] == criterio and r["distancia"] == distancia
                  and r["tam_min"] >= minimo and not r["corte_ambiguo"]]
    if not candidatas:
        return None

    def puntuacion(r):
        valor = r[criterio]
        return CRITERIOS[criterio] * valor if np.isfinite(valor) else -np.inf

    return max(candidatas, key=puntuacion)


def texto_tabla_enlaces(resultados: list[dict], k_min: int, k_max: int, n: int) -> str:
    """Tabla de la fase 1 (un clustering por enlace) con su leyenda."""
    minimo = tam_minimo_valido(n)
    con_estabilidad = any(r["estabilidad_ari_media"] is not None for r in resultados)
    cabeceras = ["Enlace", "Distancia", "k", "Tamaños", "Silhouette", "Davies-B.", "Calinski-H.", "Cofenética"]
    if con_estabilidad:
        cabeceras.append("Estab. ARI")
    cabeceras.append("Tiempo(s)")

    filas = []
    for r in resultados:
        k = r["k_mejor_silhouette_en_rango"]
        fila = [r["enlace"], r["distancia_efectiva"], str(k),
                f"{r['tam_min_cluster_k_mejor']}/{r['tam_max_cluster_k_mejor']}"
                + _aviso(r["tam_min_cluster_k_mejor"], minimo, r["corte_ambiguo_k_mejor"]),
                _num(r["silhouette_por_k"][str(k)]), _num(r["davies_bouldin_k_final"]),
                _num(r["calinski_harabasz_k_final"], 2), _num(r["correlacion_cofenetica"])]
        if con_estabilidad:
            fila.append(_num(r["estabilidad_ari_media"]))
        fila.append(f"{r['tiempo_segundos']:.2f}")
        filas.append(fila)

    lineas = ["", f"Clustering por enlace (k con mejor silhouette entre {k_min} y {k_max})",
              formatear_tabla(cabeceras, filas)]
    lineas += _leyenda(minimo,
                       any(r["tam_min_cluster_k_mejor"] < minimo for r in resultados),
                       any(r["corte_ambiguo_k_mejor"] for r in resultados),
                       any(r["distancia"] != r["distancia_efectiva"] for r in resultados))
    return "\n".join(lineas)


def texto_comparativa(resumen: list[dict], criterios: list[str], comprobar: bool, n: int) -> str:
    """Tabla de la fase 2 (mejor k por combinación y criterio) y sugerencia para config.json."""
    minimo = tam_minimo_valido(n)
    cabeceras = ["Distancia", "Enlace", "Criterio", "k", "Tamaños", "Silhouette", "Davies-B.", "Calinski-H.", "Evaluac."]
    if comprobar:
        cabeceras.append("Exhaustivo")
    filas = []
    for r in sorted(resumen, key=lambda r: (criterios.index(r["criterio"]), r["distancia"], r["enlace"])):
        fila = [r["distancia"], r["enlace"], r["criterio"], str(r["k_mejor"]),
                f"{r['tam_min']}/{r['tam_max']}" + _aviso(r["tam_min"], minimo, r["corte_ambiguo"]),
                _num(r["silhouette"]), _num(r["davies_bouldin"]), _num(r["calinski_harabasz"], 2),
                f"{r['evaluaciones_busqueda']}/{r['evaluaciones_barrido']}"]
        if comprobar:
            fila.append(f"k={r['k_exhaustivo']} ({'coincide' if r['busqueda_acierta'] else 'NO coincide'})")
        filas.append(fila)

    lineas = ["", "Mejor k por combinación y criterio", formatear_tabla(cabeceras, filas, texto=3),
              "Silhouette y Calinski-H.: mayor es mejor. Davies-B.: menor es mejor."]
    lineas += _leyenda(minimo, any(r["tam_min"] < minimo for r in resumen),
                       any(r["corte_ambiguo"] for r in resumen))

    distancias = list(dict.fromkeys(r["distancia"] for r in resumen))
    if len(distancias) > 1:
        lineas.append("Hay varias distancias: el silhouette solo es comparable entre enlaces con la misma distancia.")

    referencia = next((d for d in distancias if d in DISTANCIAS_MODELO), None)
    mejor = sugerir_mejor(resumen, criterios[0], referencia, minimo) if referencia else None
    if referencia and not mejor:
        lineas += ["", f"Sin sugerencia: todas las particiones con '{criterios[0]}' ({referencia}) tienen clusters "
                       f"de menos de {minimo} usuarios o cortes empatados. Revisa evaluaciones_k.csv "
                       f"o prueba otra representación."]
    if mejor:
        lineas += ["", f"Sugerencia según '{criterios[0]}' ({referencia}): {mejor['enlace']} con k = {mejor['k_mejor']} "
                       f"(silhouette {_num(mejor['silhouette'], 3)})."]
        if mejor["silhouette"] < SILHOUETTE_DEBIL:
            lineas.append(f"Silhouette por debajo de {SILHOUETTE_DEBIL}: los grupos son poco claros con esta "
                          f"representación. Conviene compararla con otras (--entrada ...).")
        lineas += ["Para generar el modelo, en config.json:",
                   f'  "mejor_modelo": {{"enlace": "{mejor["enlace"]}", "distancia": "{mejor["distancia"]}", '
                   f'"k": {mejor["k_mejor"]}, "archivo": "modelo.pkl"}}',
                   "y ejecuta: python -m Clustering_jerarquico modelo"]
    return "\n".join(lineas)


# Fases de la experimentación

def _ajustar_enlace(X, enlace: str, distancia: str, ks: list[int], salida: Path, plots: Path, *,
                    estabilidad_reps: int, semilla: int, generar_graficas: bool, incluir_db: bool,
                    progreso: Progreso) -> dict:
    """Ajusta un enlace, evalúa el silhouette para cada k y guarda etiquetas y gráficas. Devuelve su registro."""
    n = X.shape[0]
    modelo, X_modelo = preparar_modelo(enlace, distancia, X)

    inicio = time.perf_counter()
    resultado = modelo.fit(X_modelo)
    segundos = time.perf_counter() - inicio

    progreso.detalle(f"{enlace} · silhouette para k = {ks[0]}..{ks[-1]}")
    D_eval = matriz_de_evaluacion(resultado, X_modelo, distancia)
    sil_por_k = {str(k): silhouette_medio(D_eval, resultado.etiquetas(k)) for k in ks}
    k_mejor = max(ks, key=lambda k: sil_por_k[str(k)] if np.isfinite(sil_por_k[str(k)]) else -np.inf)
    labels = resultado.etiquetas(k_mejor)

    estabilidad = []
    if estabilidad_reps > 0:
        progreso.detalle(f"{enlace} · estabilidad con {estabilidad_reps} submuestreos")
        rng = np.random.default_rng(semilla)
        n_sub = max(k_mejor + 1, int(np.ceil(0.8 * n)))

        for _ in range(estabilidad_reps):
            idx = np.sort(rng.choice(n, size=min(n_sub, n - 1), replace=False))
            submodelo = ClusteringJerarquico(enlace=modelo.enlace, distancia=modelo.distancia, p=modelo.p)
            subresultado = submodelo.fit(X_modelo[idx])
            estabilidad.append(adjusted_rand_index(labels[idx], subresultado.etiquetas(k_mejor)))

    if generar_graficas:
        progreso.detalle(f"{enlace} · gráficas")
        sub = plots / enlace
        sub.mkdir(exist_ok=True)
        guardar_dendrograma(resultado, labels, D_eval, sub)
        guardar_silhouette(ks, [sil_por_k[str(k)] for k in ks], sub)
        guardar_proyeccion_mds(resultado.distancias_originales, labels, sub)

    # Davies-Bouldin y Calinski-Harabasz son euclídeas: con coseno se miden sobre filas normalizadas.
    X_metricas = normalizar_filas(X_modelo) if modelo.distancia == "cosine" else X_modelo

    registro = {
        "enlace": enlace,
        "distancia": distancia,
        "distancia_efectiva": modelo.distancia,
        "tiempo_segundos": round(segundos, 4),
        "correlacion_cofenetica": correlacion_cofenetica(resultado),
        "silhouette_por_k": sil_por_k,
        "k_mejor_silhouette_en_rango": int(k_mejor),
        "tam_min_cluster_k_mejor": int(np.bincount(labels).min()),
        "tam_max_cluster_k_mejor": int(np.bincount(labels).max()),
        "corte_ambiguo_k_mejor": corte_ambiguo(resultado, k_mejor),
        "estabilidad_ari_por_repeticion": estabilidad,
        "estabilidad_ari_media": float(np.mean(estabilidad)) if estabilidad else None,
        "calinski_harabasz_k_final": calinski_harabasz(X_metricas, labels),
        "davies_bouldin_k_final": None,
    }

    if incluir_db:
        try:
            registro["davies_bouldin_k_final"] = davies_bouldin(X_metricas, labels)
        except (MemoryError, ValueError) as e:
            progreso.escribir(f"Aviso: no se pudo calcular Davies-Bouldin para {enlace}: {e}")

    np.save(salida / f"etiquetas_{enlace}_k{k_mejor}.npy", labels)
    return registro


def clustering_por_enlaces(X, salida: Path, enlaces: list[str], distancia: str, k_min: int, k_max: int, *,
                           entrada: str = "", estabilidad_reps: int = 0, semilla: int = 42,
                           generar_graficas: bool = True, incluir_db: bool = False,
                           progreso: Progreso | None = None) -> list[dict]:
    """Fase 1: un clustering por enlace, silhouette en todo el rango de k, gráficas y resumen_metricas.json.

    Pre:  2 <= k_min <= k_max < n y `enlaces` contiene enlaces válidos.
    Post: escribe en `salida` las etiquetas, las gráficas y resumen_metricas.json, imprime la tabla de
          resultados y devuelve un registro por enlace. Un enlace que falla se omite con un aviso.
    """
    progreso = progreso or Progreso(visible=False)
    n = X.shape[0]
    salida.mkdir(parents=True, exist_ok=True)
    plots = salida / "plots"
    plots.mkdir(exist_ok=True)
    ks = list(range(k_min, k_max + 1))
    resultados, omitidos = [], []

    for enlace in enlaces:
        progreso.paso(f"{enlace} · ajustando el árbol")
        try:
            resultados.append(_ajustar_enlace(
                X, enlace, distancia, ks, salida, plots, estabilidad_reps=estabilidad_reps, semilla=semilla,
                generar_graficas=generar_graficas, incluir_db=incluir_db, progreso=progreso))
        except (ValueError, MemoryError) as e:
            progreso.escribir(f"Error en {enlace}: {e or type(e).__name__}. Se omite este enlace.")
            omitidos.append({"enlace": enlace, "motivo": str(e) or type(e).__name__})

    resumen = {
        "entrada": entrada,
        "instancias": int(n),
        "atributos": int(X.shape[1]),
        "nota": "El silhouette más alto guía de forma automática el número de clusters óptimo para cada enlace.",
        "omitidos": omitidos,
        "resultados": resultados,
    }
    (salida / "resumen_metricas.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if resultados:
        progreso.escribir(texto_tabla_enlaces(resultados, k_min, k_max, n))
    return resultados


def buscar_k_por_combinacion(X, salida: Path, distancias: list[str], enlaces: list[str], k_min: int, k_max: int, *,
                             criterios: list[str], busqueda: str, puntos_malla: int, comprobar: bool,
                             p: float = 2.0, entrada: str = "", progreso: Progreso | None = None) -> list[dict]:
    """Fase 2: busca el mejor k de cada distancia, enlace y criterio sin recorrer todo el rango.

    Pre:  2 <= k_min <= k_max < n y `criterios` contiene criterios de CRITERIOS.
    Post: escribe los CSV, el JSON y las gráficas en salida/experimentos/<fecha>_<busqueda>, imprime la
          comparativa con la sugerencia y devuelve el resumen. Las combinaciones inválidas o que fallan se omiten
          con un aviso (por ejemplo ward con una distancia no euclídea).
    """
    progreso = progreso or Progreso(visible=False)
    n = X.shape[0]
    carpeta = salida / "experimentos" / f"{datetime.now():%Y%m%d_%H%M%S}_{busqueda}"
    (carpeta / "plots").mkdir(parents=True, exist_ok=True)

    parametros = {
        "entrada": entrada,
        "instancias": int(n),
        "atributos": int(X.shape[1]),
        "distancias": distancias,
        "enlaces": enlaces,
        "p_minkowski": p,
        "k_min": k_min,
        "k_max": k_max,
        "criterios": criterios,
        "busqueda": busqueda,
        "puntos_malla": puntos_malla,
        "comprobacion_exhaustiva": comprobar,
    }

    todas_filas, todo_resumen, omitidas = [], [], []

    for distancia in distancias:
        for enlace in enlaces:
            etiqueta = f"{distancia} / {enlace}"
            progreso.paso(f"{etiqueta} · buscando el mejor k")
            try:
                modelo, X_modelo = preparar_modelo(enlace, distancia, X, p)
                filas, resumen = probar_combinacion(modelo, X_modelo, k_min, k_max, criterios, comprobar,
                                                    busqueda, puntos_malla, distancia, progreso, etiqueta)
            except (ValueError, MemoryError) as e:
                progreso.escribir(f"Omitida {etiqueta}: {e or type(e).__name__}")
                omitidas.append({"distancia": distancia, "enlace": enlace, "motivo": str(e) or type(e).__name__})
                continue

            todas_filas.extend(filas)
            todo_resumen.extend(resumen)

            mejores = {r["criterio"]: r["k_mejor"] for r in resumen}
            guardar_busqueda_k(filas, mejores, f"Búsqueda de k - {etiqueta}",
                               carpeta / "plots" / f"busqueda_k_{distancia}_{enlace}.png")
            guardar_resultados(carpeta, {**parametros, "omitidas": omitidas}, todas_filas, todo_resumen)

    guardar_resultados(carpeta, {**parametros, "omitidas": omitidas}, todas_filas, todo_resumen)
    if todo_resumen:
        progreso.escribir(texto_comparativa(todo_resumen, criterios, comprobar, n))
    progreso.escribir(f"\nBúsqueda de k guardada en: {carpeta.resolve()}")
    return todo_resumen
