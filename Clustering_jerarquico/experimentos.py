"""Banco de pruebas: busca el mejor k sin recorrer todo el rango y guarda silhouette, Davies-Bouldin y Calinski-Harabasz.

Se puede lanzar solo (python -m <paquete>.experimentos) o a continuación del CLI principal
(clave "ejecutar_experimentos": true en config.json).

Métodos de búsqueda (--busqueda):
    mallado  malla geométrica gruesa para localizar la zona buena y bisección dentro de ella (por defecto).
    binaria  bisección pura sobre todo el rango; solo es fiable si la curva tiene un único máximo.

Uso (desde la carpeta que contiene el paquete, igual que el CLI):
    python -m <paquete>.experimentos --k_min 2 --k_max 100
    python -m <paquete>.experimentos --busqueda binaria --comprobar
    python -m <paquete>.experimentos --distancias cosine euclidean --enlaces average complete ward --comprobar

Los valores por defecto salen de config.json; los argumentos tienen prioridad.
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from vectorizacion.matrices import cargar_matriz
from .algoritmo import ClusteringJerarquico
from .cli import cargar_configuracion, preparar_modelo
from .metricas import (silhouette_medio, davies_bouldin, calinski_harabasz,
                       correlacion_cofenetica, normalizar_filas)
from .plots import guardar_busqueda_k

# Signo con el que cada métrica se convierte en una puntuación a maximizar.
CRITERIOS = {"silhouette": 1.0, "davies_bouldin": -1.0, "calinski_harabasz": 1.0}

COLUMNAS_EVALUACIONES = [
    "distancia", "distancia_efectiva", "enlace", "k", "silhouette", "davies_bouldin",
    "calinski_harabasz", "tam_min", "tam_max",
    "visitado_silhouette", "visitado_davies_bouldin", "visitado_calinski_harabasz",
]
COLUMNAS_RESUMEN = [
    "distancia", "distancia_efectiva", "enlace", "busqueda", "criterio", "k_mejor",
    "silhouette", "davies_bouldin", "calinski_harabasz", "tam_min", "tam_max", "evaluaciones_busqueda", "evaluaciones_barrido",
    "tiempo_fit_s", "tiempo_busqueda_s", "correlacion_cofenetica",
    "k_exhaustivo", "valor_exhaustivo", "busqueda_acierta", "k_visitados",
]


def _biseccion(f, lo: int, hi: int) -> int:
    """Bisección sobre la pendiente de la curva f (a maximizar) en [lo, hi].

    Compara k=mid con k=mid+1: si mejora, el óptimo está a la derecha; si no, está en
    mid o a su izquierda. Cada paso descarta la mitad del intervalo. En empate gana el k menor.
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
    """Valores de k espaciados geométricamente (más densos donde k es pequeño)."""
    ks = np.unique(np.round(np.geomspace(k_min, k_max, puntos)).astype(int))
    return [int(k) for k in ks]


def buscar_mejor_k(puntuacion, k_min: int, k_max: int, metodo: str = "mallado",
                   puntos: int = 12) -> tuple[int, list[int]]:
    """Devuelve el mejor k y la lista de k evaluados, en orden de visita.

    'binaria': bisección pura sobre [k_min, k_max]. Con ~2·log2(k_max - k_min) evaluaciones,
    pero asume un único máximo; con varios puede quedarse en uno que no es el global.
    'mallado': evalúa una malla geométrica de `puntos` valores, toma el mejor y hace
    la bisección solo entre sus dos vecinos de la malla.
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


def probar_combinacion(modelo, X, k_min: int, k_max: int, criterios: list[str],
                       comprobar: bool, metodo: str, puntos: int, distancia_solicitada: str | None = None):
    """Ajusta el modelo una vez y busca el mejor k para cada criterio.

    Las evaluaciones se comparten entre criterios: un k ya evaluado no se recalcula.
    """
    enlace = modelo.enlace
    efectiva = modelo.distancia
    distancia = distancia_solicitada or efectiva

    inicio = time.perf_counter()
    resultado = modelo.fit(X)
    t_fit = time.perf_counter() - inicio

    D = resultado.distancias_originales
    X_db = normalizar_filas(X) if efectiva == "cosine" else X
    cache: dict[int, dict] = {}

    def evaluar(k: int) -> dict:
        if k not in cache:
            etiquetas = resultado.etiquetas(k)
            tam = np.bincount(etiquetas)
            cache[k] = {
                "k": k,
                "silhouette": silhouette_medio(D, etiquetas),
                "davies_bouldin": davies_bouldin(X_db, etiquetas),
                "calinski_harabasz": calinski_harabasz(X_db, etiquetas),
                "tam_min": int(tam.min()),
                "tam_max": int(tam.max()),
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


def main(argv=None):
    try:
        config = cargar_configuracion()
    except (FileNotFoundError, ValueError) as e:
        print(f"Error de configuración: {e}")
        return

    exp = config.get("experimentos", {})
    parser = argparse.ArgumentParser(description="Búsqueda del mejor k con silhouette, Davies-Bouldin y Calinski-Harabasz.")
    parser.add_argument("--entrada", type=Path, default=Path(config["entrada"]),
                        help="Carpeta con X.npz (dispersa) o X.npy (densa).")
    parser.add_argument("--salida", type=Path, default=Path(config["salida"]),
                        help="Carpeta base de salida para los experimentos.")
    parser.add_argument("--distancias", nargs="+", default=[config["distancia"]],
                        choices=sorted(ClusteringJerarquico.DISTANCIAS))
    parser.add_argument("--enlaces", nargs="+", default=config["enlaces"],
                        choices=sorted(ClusteringJerarquico.ENLACES))
    parser.add_argument("--p", type=float, default=2.0, help="Exponente para la distancia minkowski.")
    parser.add_argument("--k_min", type=int, default=config["k_min"])
    parser.add_argument("--k_max", type=int, default=config["k_max"])
    parser.add_argument("--criterios", nargs="+", default=exp.get("criterios", list(CRITERIOS)), choices=list(CRITERIOS))
    parser.add_argument("--busqueda", choices=["mallado", "binaria"], default=exp.get("busqueda", "mallado"))
    parser.add_argument("--puntos_malla", type=int, default=exp.get("puntos_malla", 12),
                        help="Número de valores de k de la malla gruesa (solo con --busqueda mallado).")
    parser.add_argument("--comprobar", action="store_true", default=bool(exp.get("comprobar", False)),
                        help="Evalúa todos los k del rango y compara con el resultado de la búsqueda.")
    args = parser.parse_args(argv)
    desconocidos = [c for c in args.criterios if c not in CRITERIOS]
    if desconocidos:
        parser.error(f"criterios no válidos: {desconocidos}. Usa: {list(CRITERIOS)}")

    raiz = Path(__file__).parent.parent
    try:
        X = cargar_matriz(raiz / args.entrada)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    n = X.shape[0]

    k_min = args.k_min
    k_max = min(args.k_max, n - 1)
    if k_max < args.k_max:
        print(f"Aviso: k_max se reduce a {k_max} (tiene que ser menor que el número de instancias, {n}).")
    if not 2 <= k_min <= k_max:
        print("Se requiere que: 2 <= k_min <= k_max < número de instancias.")
        return

    carpeta = raiz / args.salida / "experimentos" / f"{datetime.now():%Y%m%d_%H%M%S}_{args.busqueda}"
    (carpeta / "plots").mkdir(parents=True, exist_ok=True)

    parametros = {
        "entrada": str(args.entrada),
        "instancias": int(n),
        "atributos": int(X.shape[1]),
        "distancias": args.distancias,
        "enlaces": args.enlaces,
        "p_minkowski": args.p,
        "k_min": k_min,
        "k_max": k_max,
        "criterios": args.criterios,
        "busqueda": args.busqueda,
        "puntos_malla": args.puntos_malla,
        "comprobacion_exhaustiva": args.comprobar,
    }

    todas_filas, todo_resumen, omitidas = [], [], []

    for distancia in args.distancias:
        for enlace in args.enlaces:
            etiqueta = f"{distancia} | {enlace}"
            try:
                modelo, X_modelo = preparar_modelo(enlace, distancia, X, args.p)
            except ValueError as e:
                print(f"[{etiqueta}] omitida: {e}")
                omitidas.append({"distancia": distancia, "enlace": enlace, "motivo": str(e)})
                continue

            filas, resumen = probar_combinacion(modelo, X_modelo, k_min, k_max, args.criterios, args.comprobar,
                                                args.busqueda, args.puntos_malla, distancia)
            todas_filas.extend(filas)
            todo_resumen.extend(resumen)

            for r in resumen:
                linea = (f"[{etiqueta}] {r['criterio']}: k={r['k_mejor']} "
                         f"(sil={r['silhouette']:.4f}, db={r['davies_bouldin']:.4f}, ch={r['calinski_harabasz']:.2f}) "
                         f"con {r['evaluaciones_busqueda']} de {r['evaluaciones_barrido']} evaluaciones")
                if args.comprobar:
                    linea += f" | exhaustivo: k={r['k_exhaustivo']} ({'coincide' if r['busqueda_acierta'] else 'NO coincide'})"
                print(linea)

            mejores = {r["criterio"]: r["k_mejor"] for r in resumen}
            guardar_busqueda_k(filas, mejores, f"Búsqueda de k — {etiqueta}",
                               carpeta / "plots" / f"busqueda_k_{distancia}_{enlace}.png")
            guardar_resultados(carpeta, {**parametros, "omitidas": omitidas}, todas_filas, todo_resumen)

    guardar_resultados(carpeta, {**parametros, "omitidas": omitidas}, todas_filas, todo_resumen)
    print(f"\nResultados guardados en: {carpeta.resolve()}")


if __name__ == "__main__":
    main()
