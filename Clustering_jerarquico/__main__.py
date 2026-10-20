"""Clustering jerárquico. Se ejecuta desde la carpeta raíz del proyecto:

    python -m Clustering_jerarquico                pregunta qué hacer
    python -m Clustering_jerarquico experimentar   compara distancias y enlaces y busca el mejor k
    python -m Clustering_jerarquico modelo         entrena el "mejor_modelo" de config.json y guarda el .pkl

La configuración está en config.json; --entrada y --salida tienen prioridad sobre ella.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from vectorizacion.matrices import cargar_matriz
from Clustering_jerarquico.algoritmo import entrenar_modelo
from Clustering_jerarquico.experimentos import experimentar

RAIZ = Path(__file__).resolve().parent.parent      # carpeta del proyecto (entrada y salida van respecto a ella)


def main(argv=None) -> int:
    """Lee config.json y los argumentos y lanza el modo elegido.

    Pre:  config.json está junto a este archivo y la carpeta de entrada tiene X.npz o X.npy.
    Post: con 'experimentar' se guardan las tablas y gráficas en la salida; con 'modelo' se guarda el .pkl.
          Devuelve 0 si todo va bien.
    """
    argv = sys.argv[1:] if argv is None else argv
    config = json.loads((Path(__file__).parent / "config.json").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(prog="python -m Clustering_jerarquico", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("modo", nargs="?", choices=["experimentar", "modelo"])
    parser.add_argument("--entrada", default=config["entrada"], help="Carpeta con X.npz (dispersa) o X.npy (densa).")
    parser.add_argument("--salida", default=config["salida"], help="Carpeta de resultados.")
    args = parser.parse_args(argv)

    modo = args.modo
    if modo is None:   # sin argumentos se pregunta; con solo --entrada/--salida se experimenta
        respuesta = input("1) Experimentar  2) Generar modelo.pkl  [1]: ").strip() if not argv else "1"
        modo = "modelo" if respuesta == "2" else "experimentar"

    X = cargar_matriz(RAIZ / args.entrada).astype(np.float64)  # vienen en float32, pasamos a float64
    salida = RAIZ / args.salida
    print(f"{args.entrada}: {X.shape[0]} instancias y {X.shape[1]} atributos.\n")

    if modo == "experimentar":
        experimentar(X, config, salida)
    else:
        elegido = config["mejor_modelo"]
        modelo = entrenar_modelo(X, elegido["enlace"], elegido["distancia"], elegido["k"])
        ruta = modelo.guardar(salida / elegido.get("archivo", "modelo.pkl"))
        print(f"{modelo.resumen()}\n\nModelo guardado en: {ruta.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
