"""Curvas y umbrales de varianza de SVD sin centrar: python -m vectorizacion.analisis_svd.

Precondiciones:
    - Cada --entrada es una carpeta TF-IDF con X.npz (dispersa; X.npy no vale)
      y un usuarios.json válido (ver matrices.cargar_usuarios).
    - Cada X tiene al menos 2 filas y 2 columnas, valores finitos y varianza no nula.
    - Todas las entradas tienen los mismos usuarios en el mismo orden y nombres
      de carpeta distintos.
    - --salida no existe o está vacía.
    - --dimensiones son enteros positivos; --umbrales están en (0, 1].
    - Hay memoria para X·Xᵀ densa en float64: O(n_usuarios²)
      (unos 44 MB con 2.344 usuarios).

Postcondiciones:
    - Se crea --salida con INFORME.md, comparativa.csv, umbrales.csv,
      curva_completa.csv, resumen.json y varianza_acumulada.png/.svg.
    - La curva cubre las dimensiones 1..min(n_usuarios, n_columnas) - 1, el mismo
      límite que vectorizacion.svd. Las proporciones están en [0, 1] y la
      acumulada es creciente y nunca pasa de 1.
    - Para cada umbral se da la primera dimensión que lo alcanza, o None si no se alcanza.
    - La varianza se mide antes de normalizar L2 y sin centrar (no es PCA). No
      mide la calidad del clustering.
    - Las matrices de entrada no se modifican y no se genera ningún modelo SVD.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.linalg import eigh

from .matrices import cargar_matriz, cargar_usuarios
from .svd import huella


def curva_varianza(X):
    """Varianza de las proyecciones SVD, ordenadas por valor singular, antes de L2.

    XX^T = U diag(s^2) U^T; las coordenadas sin normalizar son U diag(s).
    No se centra X antes de descomponer: esto NO es PCA. Solo la matriz entre
    usuarios se densifica. Costes: memoria O(n_usuarios^2), diagonalización O(n^3).
    Se conserva el límite de dimensiones del comando svd: min(X.shape) - 1.
    """
    if not sparse.issparse(X) or X.ndim != 2 or min(X.shape) < 2:
        raise ValueError("Se requiere TF-IDF disperso con al menos dos filas y dos columnas.")
    if not np.isfinite(X.data).all():
        raise ValueError("La entrada contiene valores no finitos.")
    A = X.astype(np.float64).tocsr()
    gram = (A @ A.T).toarray()
    n = A.shape[0]
    # Suma de las varianzas de las columnas de X, con ddof=0 (como sklearn).
    total = float(np.trace(gram) / n - gram.mean())
    tolerancia = np.finfo(float).eps * max(float(np.trace(gram) / n), 1.0) * 100
    if total <= tolerancia:
        raise ValueError("La matriz no tiene suficiente varianza para calcular proporciones fiables.")
    valores, vectores = eigh(gram)
    valores = np.maximum(valores[::-1], 0)
    proyeccion = vectores[:, ::-1] * np.sqrt(valores)
    proporciones = np.var(proyeccion, axis=0) / total
    limite = min(A.shape) - 1
    acumulada = np.cumsum(proporciones[:limite])
    if not np.isfinite(acumulada).all() or np.any(acumulada > 1 + 1e-8):
        raise ValueError("El cálculo de la varianza no es numéricamente consistente.")
    return proporciones[:limite], np.minimum(acumulada, 1.0)


def localizar_umbral(acumulada, umbral):
    """Primera dimensión que alcanza el umbral; None si no se alcanza."""
    posicion = int(np.searchsorted(acumulada, umbral, side="left"))
    return posicion + 1 if posicion < len(acumulada) else None


def guardar_csv(ruta, filas, campos):
    with ruta.open("w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(filas)


def guardar_grafica(resultados, umbrales, salida):
    # Backend sin ventanas: permite regenerar el informe desde un terminal.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    colores = ["#1565c0", "#d85a12", "#2e7d32", "#7b1fa2"]
    for i, resultado in enumerate(resultados):
        curva = np.array(resultado["varianza_acumulada"])
        nombre = Path(resultado["entrada"]).name
        etiqueta = {"resultados_tfidf_completo": "Palabras",
                    "resultados_tfidf_caracteres4_completo": "Caracteres de longitud 4"}.get(nombre, nombre)
        color = colores[i % len(colores)]
        ax.plot(np.arange(1, len(curva) + 1), curva * 100, label=etiqueta, color=color, linewidth=2)
        for punto in resultado["umbrales"]:
            k = punto["dimensiones"]
            if k is not None:
                y = punto["varianza_acumulada"] * 100
                ax.scatter(k, y, color=color, s=30, zorder=3)
                ax.annotate(f"{k} dim.", (k, y), xytext=(-8, 12 + i * 15), textcoords="offset points",
                            ha="right", color=color, fontsize=9,
                            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
    for umbral in umbrales:
        ax.axhline(100 * umbral, color="#68737d", linestyle="--", linewidth=1,
                   label=f"Objetivo {umbral:.0%}")
    ax.set(title="TF-IDF + SVD: varianza explicada acumulada",
           xlabel="Número de dimensiones conservadas", ylabel="Varianza explicada (%)", ylim=(0, 110))
    ax.grid(alpha=0.18)
    ax.legend(loc="lower right", frameon=False)
    fig.suptitle("SVD sin centrar · antes de normalizar L2", fontsize=10, color="#59636d")
    fig.savefig(salida / "varianza_acumulada.png", dpi=180)
    fig.savefig(salida / "varianza_acumulada.svg")
    plt.close(fig)


def analizar(entradas, salida, dimensiones=(50, 100, 200, 300, 400, 500), umbrales=(0.90, 0.95)):
    salida = Path(salida)
    entradas = [Path(p) for p in entradas]
    dimensiones, umbrales = sorted(set(dimensiones)), sorted(set(umbrales))
    if not entradas or not dimensiones or any(k < 1 or int(k) != k for k in dimensiones):
        raise ValueError("Indica al menos una entrada y dimensiones enteras positivas.")
    if not umbrales or any(not 0 < u <= 1 for u in umbrales):
        raise ValueError("Los umbrales deben estar entre 0 (excluido) y 1.")
    if salida.exists() and (not salida.is_dir() or any(salida.iterdir())):
        raise ValueError("La salida debe ser una carpeta nueva o vacía.")
    if len({p.name for p in entradas}) != len(entradas):
        raise ValueError("Las carpetas de entrada deben tener nombres distintos.")

    resultados, filas_curva, filas_tabla, filas_umbral = [], [], [], []
    ids_referencia = None
    for entrada in entradas:
        X = cargar_matriz(entrada)
        usuarios = cargar_usuarios(entrada, X.shape[0])
        ids = [u["usuario"] for u in usuarios]
        if ids_referencia is not None and ids != ids_referencia:
            raise ValueError("Las entradas deben tener los mismos usuarios en el mismo orden.")
        ids_referencia = ids
        proporciones, acumulada = curva_varianza(X)
        puntos = []
        for u in umbrales:
            k = localizar_umbral(acumulada, u)
            punto = {"umbral": u, "dimensiones": k,
                     "varianza_acumulada": float(acumulada[k - 1]) if k is not None else None,
                     "varianza_dimension_anterior": float(acumulada[k - 2]) if k is not None and k > 1 else (0.0 if k == 1 else None)}
            puntos.append(punto)
            filas_umbral.append({"representacion": entrada.name, **punto})
        comparaciones = sorted({k for k in dimensiones if k <= len(acumulada)}
                               | {p["dimensiones"] for p in puntos if p["dimensiones"] is not None}
                               | {len(acumulada)})
        for k in comparaciones:
            filas_tabla.append({"representacion": entrada.name, "dimensiones": k,
                               "varianza_acumulada": float(acumulada[k - 1])})
        filas_curva.extend({"representacion": entrada.name, "dimensiones": i + 1,
                            "varianza_componente": float(v), "varianza_acumulada": float(c)}
                           for i, (v, c) in enumerate(zip(proporciones, acumulada)))
        resultados.append({"entrada": str(entrada), "shape": list(X.shape),
                           "archivos_sha256": {n: huella(entrada / n) for n in ("X.npz", "usuarios.json")},
                           "dimensiones_maximas": len(acumulada),
                           "dimensiones_solicitadas_fuera_de_rango": [k for k in dimensiones if k > len(acumulada)],
                           "varianza_acumulada": acumulada.tolist(), "umbrales": puntos})

    informe = {
        "metodo": "SVD sin centrar mediante diagonalización de X @ X.T en float64",
        "normalizacion": "Varianza medida antes de la normalización L2 de las proyecciones",
        "dimension_maxima": "min(n_usuarios, n_caracteristicas) - 1, coherente con vectorizacion.svd",
        "nota": "Mínimos de la curva calculada. El modelo randomized guardado o un nuevo ajuste pueden diferir ligeramente; comprobar su resumen. No mide calidad del clustering.",
        "dimensiones_solicitadas": dimensiones, "umbrales_solicitados": umbrales,
        "resultados": resultados,
    }
    salida.mkdir(parents=True, exist_ok=True)
    guardar_csv(salida / "curva_completa.csv", filas_curva, ["representacion", "dimensiones", "varianza_componente", "varianza_acumulada"])
    guardar_csv(salida / "comparativa.csv", filas_tabla, ["representacion", "dimensiones", "varianza_acumulada"])
    guardar_csv(salida / "umbrales.csv", filas_umbral, ["representacion", "umbral", "dimensiones", "varianza_acumulada", "varianza_dimension_anterior"])
    (salida / "resumen.json").write_text(json.dumps(informe, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lineas = ["# Análisis de varianza de TF-IDF + SVD", "",
              "SVD sin centrar, antes de normalizar L2. Los porcentajes describen varianza, no calidad del clustering.", "",
              "| Representación | Objetivo | Dimensiones mínimas | Varianza alcanzada |",
              "|---|---:|---:|---:|"]
    for f in filas_umbral:
        valor = f"{f['varianza_acumulada']:.4%}" if f["dimensiones"] is not None else "No alcanzado en el rango"
        lineas.append(f"| {f['representacion']} | {f['umbral']:.0%} | {f['dimensiones'] if f['dimensiones'] is not None else '—'} | {valor} |")
    lineas.extend(["", "## Comparación de dimensiones", "", "| Representación | Dimensiones | Varianza acumulada |", "|---|---:|---:|"])
    lineas.extend(f"| {f['representacion']} | {f['dimensiones']} | {f['varianza_acumulada']:.2%} |" for f in filas_tabla)
    lineas.extend(["", "![Curvas de varianza](varianza_acumulada.png)", "",
                   "El mínimo se busca en todas las dimensiones, no solo en las filas de esta tabla. `umbrales.csv` incluye la varianza de la dimensión anterior para comprobar el cruce.", "",
                   "Se diagonaliza la matriz entre usuarios X @ X.T sin centrarla. Las coordenadas SVD permiten calcular la varianza sin densificar todo TF-IDF. El coste de memoria crece con el cuadrado del número de usuarios.", "",
                   informe["nota"], "",
                   "Las matrices y los modelos existentes se conservan. Este análisis no genera nuevas representaciones para clustering.", ""])
    (salida / "INFORME.md").write_text("\n".join(lineas), encoding="utf-8")
    guardar_grafica(resultados, umbrales, salida)
    return informe


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--entrada", type=Path, action="append", required=True)
    parser.add_argument("--salida", type=Path, default=Path("resultados_tfidf/datos_1_a_600/resultados_analisis_svd"))
    parser.add_argument("--dimensiones", type=int, nargs="+", default=[50, 100, 200, 300, 400, 500])
    parser.add_argument("--umbrales", type=float, nargs="+", default=[0.90, 0.95])
    args = parser.parse_args(argv)
    try:
        informe = analizar(args.entrada, args.salida, args.dimensiones, args.umbrales)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    for resultado in informe["resultados"]:
        for punto in resultado["umbrales"]:
            print(f"{resultado['entrada']}: {punto['umbral']:.0%} -> {punto['dimensiones'] or 'no alcanzado'} dimensiones")
    print(f"Informe y gráficas: {args.salida.resolve()}")


if __name__ == "__main__":
    main()
