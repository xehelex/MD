import argparse
from pathlib import Path

from preprocesador import Preprocesador


def main():
    parser = argparse.ArgumentParser(description="Preprocesamiento de textos para clustering")

    parser.add_argument("--datos",default=Path("."),type=Path,help="Directorio que contiene los archivos XML")

    parser.add_argument("--chat-words",default="./config/chat_words.json",type=Path,help="Ruta del diccionario de jerga")

    args = parser.parse_args()

    argus = {
        "datos": args.datos
    }

    preprocesador = Preprocesador(argus,chat_words_path=args.chat_words)

    df = preprocesador.preprocesar_datos_clustering()

    df.to_csv("output/data_prepro.csv",index=False)

if __name__ == "__main__":
    main()