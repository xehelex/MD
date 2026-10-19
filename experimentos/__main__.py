"""Lanza el banco de pruebas: python -m experimentos --help."""

import argparse
import csv
import json
from pathlib import Path

from scipy.sparse import load_npz

from .banco import correlacion_silhouette_equilibrio, recorrer_rejilla

# Con single y complete, coseno y euclidea dan el mismo arbol (solo usan min y
# max, y los vectores van normalizados L2). Con average y ward no, asi que esas
# si van por separado.
COMBINACIONES = [
    ("single", "cosine", 2.0),
    ("complete", "cosine", 2.0),
    ("average", "cosine", 2.0),
    ("average", "euclidean", 2.0),
    ("ward", "euclidean", 2.0),
]

COLUMNAS = ["matriz", "enlace", "distancia", "k", "silhouette", "equilibrio",
            "mayor_pct", "grupos_utiles", "instancias_sueltas", "tamanos",
            "estabilidad", "inconsistencia", "cofenetica", "segundos_arbol"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, action="append", required=True,
                        help="Carpeta con X_train.npz. Repetir para comparar varias.")
    parser.add_argument("--salida", type=Path, default=Path("resultados_experimentos"))
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=20)
    parser.add_argument("--minimo", type=int, default=10,
                        help="Tamaño a partir del cual un grupo cuenta como aprovechable.")
    parser.add_argument("--estabilidad", type=int, default=0, metavar="N",
                        help="Repeticiones de remuestreo al 80%% para medir estabilidad. "
                             "0 la desactiva; cada repetición reconstruye todos los árboles.")
    args = parser.parse_args()

    if not 2 <= args.k_min <= args.k_max:
        parser.error("Tiene que cumplirse 2 <= k-min <= k-max.")

    matrices = {}
    for carpeta in args.entrada:
        archivo = carpeta / "X_train.npz"
        if not archivo.is_file():
            parser.error(f"No existe {archivo}")
        matrices[carpeta.name] = load_npz(archivo).tocsr()

    ks = list(range(args.k_min, args.k_max + 1))
    filas = recorrer_rejilla(matrices, COMBINACIONES, ks, minimo=args.minimo,
                             repeticiones=args.estabilidad)

    args.salida.mkdir(parents=True, exist_ok=True)
    with (args.salida / "rejilla.csv").open("w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=COLUMNAS)
        escritor.writeheader()
        escritor.writerows(filas)

    resumen = {
        "matrices": {n: list(X.shape) for n, X in matrices.items()},
        "combinaciones": [f"{e}/{d}" for e, d, _ in COMBINACIONES],
        "k": [args.k_min, args.k_max],
        "configuraciones_evaluadas": len(filas),
        "correlacion_silhouette_equilibrio": correlacion_silhouette_equilibrio(filas),
    }
    (args.salida / "resumen.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")

    # Dos criterios en paralelo: el k que elige el silhouette y el que propone
    # la forma del arbol. Si no coinciden, el equilibrio dice cual de los dos
    # se esta llevando un 477/1/1/1.
    print("%-30s %-9s %-10s | %3s %9s %9s | %3s %9s %9s" % (
        "matriz", "enlace", "distancia",
        "k", "silhou.", "equilib.", "k", "incons.", "equilib."))
    print("%-30s %-9s %-10s | %-23s | %s" % ("", "", "", "segun SILHOUETTE", "segun el ARBOL"))
    print("-" * 108)
    for nombre in matrices:
        for enlace, distancia, _ in COMBINACIONES:
            candidatas = [f for f in filas if f["matriz"] == nombre
                          and f["enlace"] == enlace and f["distancia"].startswith(distancia)]
            if not candidatas:
                continue
            finito = lambda v: v if v == v else -1e9
            por_sil = max(candidatas, key=lambda f: finito(f["silhouette"]))
            por_arbol = max(candidatas, key=lambda f: finito(f["inconsistencia"]))
            print("%-30s %-9s %-10s | %3d %9.3f %9.3f | %3d %9.2f %9.3f" % (
                nombre[:30], enlace, distancia,
                por_sil["k"], por_sil["silhouette"], por_sil["equilibrio"],
                por_arbol["k"], por_arbol["inconsistencia"], por_arbol["equilibrio"]))
        print()

    r = resumen["correlacion_silhouette_equilibrio"]
    print(f"{len(filas)} configuraciones evaluadas.")
    print(f"Correlacion silhouette-equilibrio: {r:.3f}")
    if r < -0.5:
        print("Negativa y fuerte: el silhouette premia las particiones degeneradas.")
    print(f"Tabla completa en: {(args.salida / 'rejilla.csv').resolve()}")


if __name__ == "__main__":
    main()
