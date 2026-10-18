import argparse
from pathlib import Path

from preprocesador import Preprocesador


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datos", required=True, help="Carpeta con los XML")
    parser.add_argument("--idioma", default="auto", choices=["auto", "english", "spanish"])
    parser.add_argument("--min_tokens", type=int, default=3)
    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--guardar_csv", action="store_true")
    args = vars(parser.parse_args())

    prep = Preprocesador(args)
    df = prep.preprocesar_datos_clustering()
    print(f"Publicaciones preprocesadas: {len(df)}")

if __name__ == "__main__":
    main()