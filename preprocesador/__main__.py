import argparse

from preprocesador import Preprocesador


def main():
    """
    Función principal que configura los argumentos de entrada, inicializa el preprocesador y ejecuta
    el preprocesamiento de los datos o, si se indica --analisis, únicamente el análisis exploratorio.

    Precondiciones:
        - Ejecución desde la línea de comandos con los argumentos requeridos (ej. --datos).
        - Existencia del módulo y clase Preprocesador accesibles en el entorno.

    Postcondiciones:
        - Sin --analisis: se procesa el DataFrame de publicaciones y se imprime el total resultante.
        - Con --analisis: se guarda el análisis en output/analisis.txt y se muestra el gráfico de idiomas.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--datos", required=True, help="Carpeta con los XML, o CSV exportado (output/data.csv)")
    parser.add_argument("--analisis", action="store_true", help="Solo análisis exploratorio (sin preprocesar)")
    parser.add_argument("--guardar_fig", help="Con --analisis: ruta donde guardar el gráfico de idiomas")
    parser.add_argument("--idioma", default="auto", choices=["auto", "english", "spanish"])
    parser.add_argument("--min_tokens", type=int, default=3)
    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--guardar_csv", action="store_true")
    args = vars(parser.parse_args())

    prep = Preprocesador(args)
    if args["analisis"]:
        prep.analisis_exploratorio(guardar_fig=args["guardar_fig"])
    else:
        df = prep.preprocesar_datos_clustering()
        print(f"Publicaciones preprocesadas: {len(df)}")

if __name__ == "__main__":
    main()
