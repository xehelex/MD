"""Consulta un texto contra el modelo: python -m experimentos.consultar --help."""

import argparse
from pathlib import Path
import sys

from Clustering_jerarquico.algoritmo import ModeloEntrenado

from .nueva_instancia import Clasificador, es_densa


def main(argv=None):
    """Consulta un texto contra el modelo y lo escribe por pantalla.

    Pre:  --entrada es la carpeta con la que se entreno --modelo, y el texto
          llega por argumento o por teclado.
    Post: imprime el cluster y los vecinos mas parecidos, y guarda el PNG si
          se pidio --grafico. Devuelve 1 si el texto no se puede procesar.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("texto", nargs="?", help="Texto a consultar. Sin esto, lo pide por teclado.")
    parser.add_argument("--entrada", type=Path, required=True,
                        help="Carpeta de la representacion con la que se entreno el modelo.")
    parser.add_argument("--modelo", type=Path, default=Path("resultados_clustering_completo/modelo.pkl"))
    parser.add_argument("--textos", type=Path, default=Path("output/data_preprocesado.csv"),
                        help="CSV del preproceso, para enseñar el texto de los vecinos.")
    parser.add_argument("--chat-words", type=Path, default=Path("preprocesador/chat_words.json"))
    parser.add_argument("--vecinos", type=int, default=5)
    parser.add_argument("--grafico", type=Path, help="Guarda un PNG con la nube y el texto marcado.")
    args = parser.parse_args(argv)

    if args.texto is None:          # sin argumento lo pedimos; con "" es que lo mando vacio
        try:
            args.texto = input("Texto a consultar: ")
        except EOFError:
            parser.error("No hay texto que consultar.")
    texto = args.texto.strip()
    if not texto:
        parser.error("No has escrito nada.")

    if not args.modelo.is_file():
        parser.error(f"No existe {args.modelo}. Generalo con: python -m Clustering_jerarquico modelo")

    modelo = ModeloEntrenado.cargar(args.modelo)

    # Roberta quiere el texto natural, asi que en la densa el preprocesador
    # sobra y de paso nos ahorramos cargar nltk.
    prep = None
    if not es_densa(args.entrada):
        from preprocesador import Preprocesador

        prep = Preprocesador({"datos": "."}, chat_words_path=args.chat_words)

    clasificador = Clasificador(modelo, args.entrada,
                                csv_textos=args.textos if args.textos.is_file() else None,
                                preprocesador=prep)
    try:
        resultado = clasificador.consultar(texto, vecinos=args.vecinos)
    except ValueError as error:
        print(f"No se ha podido procesar: {error}")
        return 1

    print(f"\n{modelo.enlace} / {modelo.distancia} / k = {modelo.k}")
    if resultado.ruta == "vecinos":
        print(f"Sin el reductor UMAP ({clasificador.sin_reductor}): se asigna por vecinos, "
              "que acierta menos. Hace falta el mismo Python con el que se genero.")
    cae = f"El texto cae en el cluster {resultado.cluster}, con {resultado.tamano_cluster} usuarios."
    if resultado.votos:
        cae += f" ({resultado.votos[0]} de {resultado.votos[1]} vecinos)"
    print(cae)
    aviso = resultado.aviso()
    print(f"AVISO: {aviso}\n" if aviso else "")
    print(f"{'usuario':14} {'dist':>7}  texto")
    print("-" * 78)
    for vecino in resultado.vecinos:
        print(f"{vecino.usuario:14} {vecino.distancia:7.3f}  {vecino.texto[:52]}")

    if args.grafico:
        print(f"\nGrafico: {clasificador.grafico(resultado, args.grafico).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
