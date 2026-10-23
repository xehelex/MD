import argparse

from preprocesador import Preprocesador


def main():
    """
    Función principal que configura los argumentos de entrada, inicializa el preprocesador y ejecuta el flujo de preprocesamiento de los datos.

    Precondiciones:
        - Ejecución desde la línea de comandos con los argumentos requeridos (ej. --datos).
        - Existencia del módulo y clase Preprocesador accesibles en el entorno.

    Postcondiciones:
        - Se genera y procesa el DataFrame de publicaciones mediante el preprocesador y se imprime por pantalla el número total de publicaciones resultantes.
    """
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