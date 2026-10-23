"""Ejecuta todas las representaciones preparadas de un conjunto desde la raíz de MD.

    python -m experimentos.todas                              # 600 primeros (8 entradas)
    python -m experimentos.todas --conjunto datos_completos   # 2.348 usuarios (6 entradas)
"""

import argparse
import sys
from pathlib import Path

from .__main__ import main as experimentar


RAIZ_TFIDF = Path("resultados_tfidf")

# Los históricos (train/test de la primera versión) solo existen para los 600.
CONJUNTOS = {
    "datos_1_a_600": (
        "resultados_tfidf_historico_alineado",
        "resultados_tfidf_caracteres4_historico_alineado",
        "resultados_tfidf_completo",
        "resultados_tfidf_caracteres4_completo",
        "resultados_tfidf_completo_svd100",
        "resultados_tfidf_caracteres4_completo_svd100",
        "resultados_tfidf_completo_svd95pct",
        "resultados_tfidf_caracteres4_completo_svd95pct",
    ),
    "datos_completos": (
        "resultados_tfidf_completo",
        "resultados_tfidf_caracteres4_completo",
        "resultados_tfidf_completo_svd100",
        "resultados_tfidf_caracteres4_completo_svd100",
        "resultados_tfidf_completo_svd95pct",
        "resultados_tfidf_caracteres4_completo_svd95pct",
    ),
}


def entradas(conjunto):
    """Rutas de las entradas de un conjunto, en el orden del lanzador.

    Pre:  conjunto es una clave de CONJUNTOS.
    Post: lista de rutas dentro de resultados_tfidf/<conjunto>/, sin comprobar que existan.
    """
    return [RAIZ_TFIDF / conjunto / carpeta for carpeta in CONJUNTOS[conjunto]]


def main(argv=None):
    """Lanza experimentos (python -m experimentos) con todas las entradas de un conjunto.

    Pre:  argv son opciones de experimentos más --conjunto (por defecto datos_1_a_600) y
          las carpetas del conjunto existen.
    Post: resultados en resultados_experimentos_todas_<conjunto>, salvo que argv traiga otra
          --salida; devuelve lo mismo que experimentos.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--conjunto", choices=tuple(CONJUNTOS), default="datos_1_a_600")
    args, resto = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if "-h" in resto or "--help" in resto:   # la ayuda la pinta experimentos, que no conoce --conjunto
        print(f"Opción propia de experimentos.todas:\n  --conjunto {{{','.join(CONJUNTOS)}}}"
              "  conjunto de representaciones (por defecto datos_1_a_600)\n")
    argumentos = ["--salida", f"resultados_experimentos_todas_{args.conjunto}"]
    for carpeta in entradas(args.conjunto):
        argumentos.extend(["--entrada", str(carpeta)])
    argumentos.extend(resto)
    return experimentar(argumentos)


if __name__ == "__main__":
    main()
