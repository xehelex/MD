"""Ejecuta las ocho representaciones preparadas desde la raíz de MD."""

import sys

from .__main__ import main as experimentar


ENTRADAS = (
    "resultados_tfidf_historico_alineado",
    "resultados_tfidf_caracteres4_historico_alineado",
    "resultados_tfidf_completo",
    "resultados_tfidf_caracteres4_completo",
    "resultados_tfidf_completo_svd100",
    "resultados_tfidf_caracteres4_completo_svd100",
    "resultados_tfidf_completo_svd95pct",
    "resultados_tfidf_caracteres4_completo_svd95pct",
)


def main(argv=None):
    argumentos = ["--salida", "resultados_experimentos_todas"]
    for carpeta in ENTRADAS:
        argumentos.extend(["--entrada", carpeta])
    argumentos.extend(sys.argv[1:] if argv is None else argv)
    return experimentar(argumentos)


if __name__ == "__main__":
    main()
