"""Lanza el banco de pruebas: python -m experimentos --help."""

import argparse
import csv
import json
from pathlib import Path

from scipy.sparse import load_npz

from .banco import correlacion_silhouette_equilibrio, recorrer_rejilla

# Coseno y euclidea dan el MISMO arbol con single y complete, porque solo usan
# minimos y maximos y los vectores van normalizados L2. Con average y ward no,
# asi que esas dos si se prueban por separado.
COMBINACIONES = [
    ("single", "cosine", 2.0),
    ("complete", "cosine", 2.0),
    ("average", "cosine", 2.0),
    ("average", "euclidean", 2.0),
    ("ward", "euclidean", 2.0),
]

COLUMNAS = ["matriz", "enlace", "distancia", "k", "silhouette", "equilibrio",
            "mayor_pct", "grupos_utiles", "instancias_sueltas", "tamanos",
            "estabilidad", "cofenetica", "segundos_arbol"]


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

    # Por pantalla, solo lo que se mira de un vistazo: el mejor k de cada
    # combinacion segun silhouette, con su equilibrio al lado para que se vea
    # si esa eleccion tiene sentido o es una particion degenerada.
    cabecera = ["matriz", "enlace", "distancia", "k", "silhouette", "equilibrio", "mayor%"]
    plantilla = "%-34s %-9s %-10s %3s %10s %10s %8s"
    if args.estabilidad:
        cabecera.append("estabil.")
        plantilla += " %9s"
    print(plantilla % tuple(cabecera))
    print("-" * (102 if args.estabilidad else 92))
    for nombre in matrices:
        for enlace, distancia, _ in COMBINACIONES:
            candidatas = [f for f in filas if f["matriz"] == nombre
                          and f["enlace"] == enlace and f["distancia"].startswith(distancia)]
            if not candidatas:
                continue
            mejor = max(candidatas, key=lambda f: f["silhouette"] if f["silhouette"] == f["silhouette"] else -1e9)
            valores = [nombre[:34], enlace, distancia, mejor["k"],
                       mejor["silhouette"], mejor["equilibrio"], mejor["mayor_pct"]]
            linea = "%-34s %-9s %-10s %3d %10.3f %10.3f %7.1f%%"
            if args.estabilidad:
                valores.append(mejor["estabilidad"])
                linea += " %9.3f"
            print(linea % tuple(valores))
        print()

    r = resumen["correlacion_silhouette_equilibrio"]
    print(f"{len(filas)} configuraciones evaluadas.")
    print(f"Correlacion silhouette-equilibrio: {r:.3f}")
    if r < -0.5:
        print("Negativa y fuerte: el silhouette premia las particiones degeneradas.")
    print(f"Tabla completa en: {(args.salida / 'rejilla.csv').resolve()}")


if __name__ == "__main__":
    main()
