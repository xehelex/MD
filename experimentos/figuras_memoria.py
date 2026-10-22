"""Tablas y figuras de la memoria: compara las vectorizaciones y elige el modelo.

    python -m experimentos.figuras_memoria [--conjunto datos_completos] [--salida resultados_memoria]
                                           [--estabilidad 5] [--bootstrap 30]

Compara todas las representaciones TF-IDF/SVD del conjunto y las de resultados_roberta/<conjunto>/.
El rango de k, la búsqueda (mallado) y sus puntos se leen de Clustering_jerarquico/config.json.
Todas las particiones se miden con la distancia coseno, sea cual sea la del árbol: el silhouette de
dos distancias distintas no se puede comparar. En las reducidas con UMAP el silhouette se mide en el
espacio del modelo sin reducir, porque UMAP deforma las distancias y lo infla.

Función objetivo: J = silhouette x equilibrio, solo si todos los grupos tienen al menos `minimo`
usuarios. Después se exige estabilidad (ARI al 80 %) >= 0,5. El silhouette solo premia los 596/2.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vectorizacion.matrices import cargar_matriz
from Clustering_jerarquico.algoritmo import ClusteringJerarquico
from Clustering_jerarquico.experimentos import buscar_k
from Clustering_jerarquico.metricas import (calinski_harabasz, correlacion_cofenetica, davies_bouldin,
                                             normalizar_filas, silhouette_medio)
from Clustering_jerarquico.plots import guardar_dendrograma, guardar_mds, guardar_metricas_k

from .banco import equilibrio, estabilidad_por_k, grupos_utiles
from .bootstrap import bootstrap
from .todas import CONJUNTOS, entradas

CONFIG = Path(__file__).resolve().parent.parent / "Clustering_jerarquico" / "config.json"
NOMBRES = {
    "resultados_tfidf_historico_alineado": "TF-IDF palabras (hist.)",
    "resultados_tfidf_caracteres4_historico_alineado": "TF-IDF caracteres (hist.)",
    "resultados_tfidf_completo": "TF-IDF palabras",
    "resultados_tfidf_caracteres4_completo": "TF-IDF caracteres",
    "resultados_tfidf_completo_svd100": "SVD 100 palabras",
    "resultados_tfidf_caracteres4_completo_svd100": "SVD 100 caracteres",
    "resultados_tfidf_completo_svd95pct": "SVD 95 % palabras",
    "resultados_tfidf_caracteres4_completo_svd95pct": "SVD 95 % caracteres",
}
MODELOS_DENSOS = {"roberta": "RoBERTa-large", "all-roberta": "all-RoBERTa", "multilingue": "MiniLM multilingüe"}
ENLACES = ("single", "complete", "average", "ward")
COLOR_ENLACE = {"single": "#8c8c8c", "complete": "#e08a2c", "average": "#7b5ea7", "ward": "#2b6cb0"}
ARI_MINIMO = 0.5

plt.rcParams.update({"font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14})


def representaciones(conjunto: str) -> list[dict]:
    """Representaciones del conjunto: las TF-IDF/SVD de todas.py y las de RoBERTa que existan.

    Pre:  conjunto está en CONJUNTOS.
    Post: lista de {nombre, preprocesado, carpeta, referencia}. referencia es la carpeta sin reducir
          con la que se mide el silhouette (solo en las reducidas con UMAP), o None.
    """
    lista = [{"nombre": NOMBRES[c.name], "preprocesado": "completo", "carpeta": c, "referencia": None}
             for c in entradas(conjunto)]
    for modelo, etiqueta in MODELOS_DENSOS.items():
        base = Path("resultados_roberta") / conjunto / modelo
        for carpeta in sorted(p.parent for p in base.glob("*/X.npy")):
            if carpeta.name == "sin_reduccion":
                nombre, referencia = etiqueta, None
            else:   # PCA_umap_<varianza>_<min_dist>
                nombre, referencia = f"{etiqueta} + UMAP (md {carpeta.name.split('_')[-1]})", base / "sin_reduccion"
            lista.append({"nombre": nombre, "preprocesado": "ligero", "carpeta": carpeta, "referencia": referencia})
    return lista


def medir(X, resultado, k: int, minimo: int, D_ref=None) -> dict:
    """Métricas de un corte, todas con la distancia coseno.

    Pre:  resultado es el árbol ajustado con X (filas normalizadas, distancia coseno) y 2 <= k < n.
          D_ref es la matriz de distancias con la que medir el silhouette (None = la del árbol).
    Post: diccionario con silhouette (en D_ref), silhouette_propio (en el espacio del árbol), equilibrio,
          Davies-Bouldin, Calinski-Harabasz, tamaños y J (-inf si algún grupo tiene menos de minimo usuarios).
    """
    et = resultado.etiquetas(k)
    tam = np.sort(np.bincount(et))[::-1]
    _, sueltas = grupos_utiles(et, minimo)
    propio = silhouette_medio(resultado.distancias_originales, et)
    s = propio if D_ref is None else silhouette_medio(D_ref, et)
    e = equilibrio(et)
    return {"k": k, "silhouette": s, "silhouette_propio": propio, "equilibrio": e,
            "J": s * e if sueltas == 0 else -np.inf,
            "davies_bouldin": davies_bouldin(X, et), "calinski_harabasz": calinski_harabasz(X, et),
            "tamanos": "/".join(str(t) for t in tam[:6])}


def guardar_csv(ruta: Path, filas: list[dict]):
    """Pre: filas no está vacía y todas tienen las mismas claves.  Post: CSV escrito en ruta."""
    with ruta.open("w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)


def probar(X, enlace: str, config: dict, minimo: int, repeticiones: int, D_ref=None) -> tuple:
    """Árbol de un enlace, mejor k por mallado (con J y con el silhouette solo) y comprobación exhaustiva.

    Pre:  X con filas normalizadas; config tiene k_min, k_max, busqueda y puntos_malla;
          D_ref como en medir().
    Post: (árbol, {k: métricas} de todos los k, fila de resumen).
    """
    inicio = time.perf_counter()
    resultado = ClusteringJerarquico(enlace, "cosine").fit(X)
    segundos = time.perf_counter() - inicio
    k_min, k_max = config["k_min"], min(config["k_max"], X.shape[0] - 1)
    cache, tocados = {}, set()

    def criterio(clave):
        """Pre: clave es una métrica de medir().  Post: función k -> valor de esa métrica (-inf si no es finito)."""
        def f(k):
            tocados.add(k)
            if k not in cache:
                cache[k] = medir(X, resultado, k, minimo, D_ref)
            return cache[k][clave] if np.isfinite(cache[k][clave]) else -np.inf
        return f

    k_j = buscar_k(criterio("J"), k_min, k_max, config["busqueda"], config["puntos_malla"])
    evaluados = len(tocados)
    k_s = buscar_k(criterio("silhouette"), k_min, k_max, config["busqueda"], config["puntos_malla"])
    k_j_todos = buscar_k(criterio("J"), k_min, k_max, "completa")      # comprobación: todos los k
    estable = estabilidad_por_k(X, resultado, enlace, "cosine", 2.0, sorted({k_j, k_s}), repeticiones)
    fila = {"enlace": enlace, **cache[k_j], "estabilidad": estable[k_j],
            "k_silhouette": k_s, "tamanos_silhouette": cache[k_s]["tamanos"],
            "silhouette_solo": cache[k_s]["silhouette"], "estabilidad_silhouette": estable[k_s],
            "cofenetica": correlacion_cofenetica(resultado), "k_evaluados_mallado": evaluados,
            "k_exhaustivo": k_j_todos, "segundos_arbol": round(segundos, 3)}
    return resultado, cache, fila


# region Figuras

def figura_trampa(puntos: list[dict], ganador: dict, ruta: Path):
    """Silhouette frente a equilibrio de todas las particiones, con las curvas de igual J.

    Pre:  puntos tiene silhouette, equilibrio, enlace, k y tamanos; ganador tiene además J y vectorizacion.
    Post: imagen guardada en ruta.
    """
    s = np.array([p["silhouette"] for p in puntos])
    e = np.array([p["equilibrio"] for p in puntos])
    r = np.corrcoef(s, e)[0, 1]
    techo = max(0.55, s.max() + 0.12)          # sitio arriba para las etiquetas
    fig, ax = plt.subplots(figsize=(10, 6.5))
    malla = np.linspace(0.02, 1, 200)
    for j in (0.1, 0.2, ganador["J"]):
        ax.plot(malla, j / malla, color="#bbbbbb", linestyle="--", linewidth=1, zorder=0)
        y_j = 0.68 * techo                       # a media altura, lejos de las anotaciones de arriba
        ax.text(j / y_j + 0.012, y_j, f"J = {j:.2f}", color="#777777", fontsize=11, va="center")
    for enlace in ENLACES:
        m = np.array([p["enlace"] == enlace for p in puntos])
        ax.scatter(e[m], s[m], s=18, alpha=0.5, color=COLOR_ENLACE[enlace], label=enlace, zorder=2)
    alto = max(puntos, key=lambda p: p["silhouette"])
    ax.annotate(f"Mayor silhouette: {alto['enlace']}, k = {alto['k']}\nreparto {alto['tamanos']}",
                (alto["equilibrio"], alto["silhouette"]), xytext=(0.12, 0.93), textcoords="axes fraction",
                fontsize=12, arrowprops={"arrowstyle": "->", "color": "#555555"})
    ax.scatter(ganador["equilibrio"], ganador["silhouette"], s=380, marker="*", color="#d62728",
               edgecolor="black", zorder=3, label="modelo elegido")
    ax.annotate(f"{ganador['vectorizacion']}\n{ganador['enlace']}, k = {ganador['k']} ({ganador['tamanos']})",
                (ganador["equilibrio"], ganador["silhouette"]), xytext=(0.6, 0.9), textcoords="axes fraction",
                fontsize=12, arrowprops={"arrowstyle": "->", "color": "#d62728"})
    ax.set(xlim=(0, 1.02), ylim=(min(-0.05, s.min() - 0.02), techo), xlabel="Equilibrio (1 = grupos iguales)",
           ylabel="Silhouette (coseno)",
           title=f"El silhouette premia los repartos desequilibrados (r = {r:.2f})")
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def figura_vectorizaciones(mejores: list[dict], sin_valida: list[str], ruta: Path):
    """Mejor J de cada vectorización, de mayor a menor, con el enlace, k y estabilidad al lado.

    Pre:  mejores tiene una fila por vectorización (J, vectorizacion, enlace, k, estabilidad);
          sin_valida son las vectorizaciones sin ninguna partición válida, que se dibujan abajo.
    Post: imagen guardada en ruta.
    """
    mejores = [{"vectorizacion": v, "J": 0.0} for v in sin_valida] + sorted(mejores, key=lambda f: f["J"])
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colores = ["#d62728" if i == len(mejores) - 1 else "#b0b0b0" for i in range(len(mejores))]
    ax.barh([f["vectorizacion"] for f in mejores], [f["J"] for f in mejores], color=colores)
    for i, f in enumerate(mejores):
        if "enlace" not in f:
            ax.text(0.004, i, "ninguna partición estable (ARI < 0.5)", va="center", fontsize=11, color="#777777")
            continue
        ax.text(f["J"] + 0.004, i, f"{f['J']:.3f}  ·  {f['enlace']}, k = {f['k']}, ARI {f['estabilidad']:.2f}",
                va="center", fontsize=11)
    ax.set(xlim=(0, max(f["J"] for f in mejores) * 1.75), xlabel="J = silhouette × equilibrio (mayor es mejor)",
           title="Mejor partición de cada vectorización")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)

# endregion


def terminos_por_grupo(etiquetas, conjunto: str, cuantos: int = 15) -> dict:
    """Palabras con más TF-IDF medio en cada grupo respecto al resto (TF-IDF de palabras).

    Pre:  etiquetas sigue el orden de usuarios de resultados_tfidf_completo del conjunto.
    Post: {grupo: [palabras]}.
    """
    import joblib
    carpeta = Path("resultados_tfidf") / conjunto / "resultados_tfidf_completo"
    X = cargar_matriz(carpeta)
    vocabulario = joblib.load(carpeta / "vectorizador.joblib").get_feature_names_out()
    salida = {}
    for g in np.unique(etiquetas):
        dentro = np.asarray(X[etiquetas == g].mean(axis=0)).ravel()
        fuera = np.asarray(X[etiquetas != g].mean(axis=0)).ravel()
        salida[int(g)] = vocabulario[np.argsort(dentro - fuera)[::-1][:cuantos]].tolist()
    return salida


def main(argv=None):
    """Genera las tablas y las figuras de la memoria.

    Pre:  se ejecuta desde la raíz de MD y existen las carpetas del conjunto en resultados_tfidf/
          (y, si las hay, en resultados_roberta/).
    Post: CSV, PNG y resumen.json guardados en la carpeta de salida (por defecto resultados_memoria).
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--conjunto", choices=tuple(CONJUNTOS), default="datos_completos")
    parser.add_argument("--salida", type=Path, default=Path("resultados_memoria"))
    parser.add_argument("--minimo", type=int, default=10)
    parser.add_argument("--estabilidad", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=30, help="Remuestreos bootstrap del modelo ganador.")
    args = parser.parse_args(argv)
    args.salida.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))

    resumen, puntos, arboles, referencias = [], [], {}, {}
    lista = representaciones(args.conjunto)
    for rep in lista:
        X = normalizar_filas(cargar_matriz(rep["carpeta"]).astype(np.float64))
        D_ref = None
        if rep["referencia"] is not None:   # UMAP: silhouette en el espacio del modelo sin reducir
            if rep["referencia"] not in referencias:
                X_ref = normalizar_filas(cargar_matriz(rep["referencia"]).astype(np.float64))
                referencias[rep["referencia"]] = np.maximum(1.0 - X_ref @ X_ref.T, 0.0)
                np.fill_diagonal(referencias[rep["referencia"]], 0.0)
            D_ref = referencias[rep["referencia"]]
        for enlace in ENLACES:
            resultado, cache, fila = probar(X, enlace, config, args.minimo, args.estabilidad, D_ref)
            resumen.append({"vectorizacion": rep["nombre"], "preprocesado": rep["preprocesado"],
                            "atributos": X.shape[1], **fila})
            puntos += [{"vectorizacion": rep["nombre"], "enlace": enlace, **m} for m in cache.values()]
            arboles[(rep["nombre"], enlace)] = (X, resultado, D_ref)
        print(f"{rep['nombre']}: hecho", flush=True)
    guardar_csv(args.salida / "resumen_vectorizaciones.csv", resumen)
    guardar_csv(args.salida / "particiones.csv", puntos)

    # Ranking: configuraciones válidas (grupos >= minimo y ARI >= 0,5) de mayor a menor J
    ranking = sorted((f for f in resumen if np.isfinite(f["J"]) and f["estabilidad"] >= ARI_MINIMO),
                     key=lambda f: (round(f["J"], 9), f["estabilidad"]), reverse=True)   # empate: el más estable
    guardar_csv(args.salida / "ranking.csv", ranking)
    nombres = [rep["nombre"] for rep in lista]
    por_vectorizacion = [next(f for f in ranking if f["vectorizacion"] == v)
                         for v in nombres if any(f["vectorizacion"] == v for f in ranking)]
    guardar_csv(args.salida / "mejor_por_vectorizacion.csv", por_vectorizacion)
    ganador = ranking[0]
    aciertos = sum(f["k"] == f["k_exhaustivo"] for f in resumen)
    print(f"\nMallado: {aciertos}/{len(resumen)} aciertos frente a probar todos los k; "
          f"{np.mean([f['k_evaluados_mallado'] for f in resumen]):.1f} k evaluados de media")
    print(f"Ganador: {ganador['vectorizacion']}, {ganador['enlace']}, k = {ganador['k']}, J = {ganador['J']:.3f}")

    X, resultado, D_ref = arboles[(ganador["vectorizacion"], ganador["enlace"])]
    boot = bootstrap(X, ganador["enlace"], "cosine", k=ganador["k"], repeticiones=args.bootstrap)
    print(boot.resumen())

    # Figuras: la comparación y las tres de plots.py para el modelo elegido
    figura_trampa(puntos, ganador, args.salida / "fig_silhouette_vs_equilibrio.png")
    sin_valida = [v for v in nombres if v not in {f["vectorizacion"] for f in por_vectorizacion}]
    figura_vectorizaciones(por_vectorizacion, sin_valida, args.salida / "fig_mejor_por_vectorizacion.png")
    curva = [medir(X, resultado, k, args.minimo, D_ref) for k in range(config["k_min"], config["k_max"] + 1)]
    mejores_k = {"silhouette": max(curva, key=lambda f: f["silhouette"])["k"],
                 "davies_bouldin": min(curva, key=lambda f: f["davies_bouldin"])["k"],
                 "calinski_harabasz": max(curva, key=lambda f: f["calinski_harabasz"])["k"]}
    guardar_metricas_k(curva, mejores_k, f"{ganador['vectorizacion']} / {ganador['enlace']} / coseno",
                       args.salida / "fig_metricas_por_k.png")
    etiquetas = resultado.etiquetas(ganador["k"])
    guardar_dendrograma(resultado, etiquetas, args.salida / "fig_dendrograma.png")
    guardar_mds(resultado.distancias_originales, etiquetas, args.salida / "fig_mds.png")

    grupos = [{"grupo": g, "usuarios": int(np.sum(etiquetas == g)), "palabras": palabras}
              for g, palabras in terminos_por_grupo(etiquetas, args.conjunto).items()]
    for g in grupos:
        print(f"Grupo {g['grupo']} ({g['usuarios']} usuarios): {', '.join(g['palabras'])}")

    s = np.array([p["silhouette"] for p in puntos])
    e = np.array([p["equilibrio"] for p in puntos])
    (args.salida / "resumen.json").write_text(json.dumps({
        "conjunto": args.conjunto, "usuarios": int(X.shape[0]), "k": [config["k_min"], config["k_max"]],
        "representaciones": len(lista), "configuraciones": len(resumen), "particiones": len(puntos),
        "correlacion_silhouette_equilibrio": float(np.corrcoef(s, e)[0, 1]),
        "mallado": {"aciertos": aciertos, "total": len(resumen),
                    "k_evaluados_media": float(np.mean([f["k_evaluados_mallado"] for f in resumen])),
                    "k_posibles": config["k_max"] - config["k_min"] + 1},
        "ganador": ganador, "bootstrap": {"media": boot.media, "intervalo": boot.intervalo(),
                                          "repeticiones": args.bootstrap},
        "mejores_k_por_metrica": mejores_k, "grupos": grupos, "top10": ranking[:10],
    }, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"\nTablas y figuras en: {args.salida.resolve()}")


if __name__ == "__main__":
    main()
