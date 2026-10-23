# Menu para ejecutar las partes del proyecto. Se lanza desde la carpeta MD: python main.py
import json
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).parent
CONFIG = RAIZ / "Clustering_jerarquico" / "config.json"
CONFIG_DENSA = RAIZ / "vectorizacion_densa" / "config.json"
CONJUNTOS = ["datos_1_a_600", "datos_completos"]
MODELOS_DENSOS = ["roberta", "all-roberta", "multilingue"]     # los de MODELOS en vectorizacion_densa


def ejecutar(*args):
    """Lanza "python -m args" desde la carpeta del proyecto.

    Pre:  args[0] es un módulo del proyecto (por ejemplo "vectorizacion").
    Post: el módulo ha terminado o se ha cancelado con Ctrl+C. Devuelve True si terminó bien.
    """
    print("\n> python -m", " ".join(args), "\n", flush=True)
    try:
        resultado = subprocess.run([sys.executable, "-m", *args], cwd=RAIZ)
        if resultado.returncode != 0:
            print("\nHa terminado con error")
        return resultado.returncode == 0
    except KeyboardInterrupt:
        print("\nCancelado")
        return False


def pedir_numero(maximo):
    """Pide una opción por teclado hasta que sea válida.

    Pre:  maximo >= 0.
    Post: devuelve un número entre 0 y maximo.
    """
    while True:
        opcion = input("Opción: ").strip()
        if opcion.isdigit() and int(opcion) <= maximo:
            return int(opcion)
        print("Opción no válida")


def elegir(titulo, opciones):
    """Enseña las opciones numeradas y deja elegir una.

    Pre:  opciones no está vacía.
    Post: devuelve la opción elegida, o None si se elige 0.
    """
    print("\n" + titulo)
    for i, opcion in enumerate(opciones, 1):
        print(f"{i}) {opcion}")
    print("0) Volver")
    numero = pedir_numero(len(opciones))
    if numero == 0:
        return None
    return opciones[numero - 1]


def preprocesar():
    """Pide la carpeta de los XML y lanza el preprocesador.

    Pre:  la carpeta que se escribe tiene los XML de CLEF.
    Post: CSV guardados en output/. Si no se escribe nada, no hace nada.
    """
    carpeta = input("Carpeta con los XML: ").strip().strip('"')
    if carpeta == "":
        return
    ejecutar("preprocesador", "--datos", carpeta, "--guardar_csv")


def analisis():
    """Análisis exploratorio de los datos sin preprocesar (preprocesador --analisis).

    Pre:  se escribe la carpeta con los XML de CLEF o el output/data.csv que guarda el preprocesador.
    Post: informe guardado en output/analisis.txt y enseñado por pantalla, y gráfico de idiomas
          en una ventana (y en un PNG si se escribe una ruta). Si no se escribe nada, no hace nada.
    """
    datos = input("Carpeta con los XML o output/data.csv: ").strip().strip('"')
    if datos == "":
        return
    args = ["preprocesador", "--datos", datos, "--analisis"]
    figura = input("PNG para guardar el gráfico de idiomas (vacío para no guardarlo): ").strip().strip('"')
    if figura != "":
        args += ["--guardar_fig", figura]

    informe = RAIZ / "output" / "analisis.txt"
    if ejecutar(*args) and informe.exists():
        print("\n" + informe.read_text(encoding="utf-8"))
        print(f"Informe guardado en {informe.relative_to(RAIZ).as_posix()}")


def tfidf(representacion):
    """Pregunta las opciones del TF-IDF y lanza la vectorización.

    Pre:  representacion es "palabras" o "caracteres4" y existe output/data_preprocesado.csv.
    Post: matriz TF-IDF guardada en la carpeta de salida.
    """
    salida = input("Carpeta de salida (vacía para la de por defecto): ").strip()
    if salida == "":
        salida = "resultados_tfidf/nuevos/resultados_tfidf_" + representacion
    args = ["vectorizacion", "--representacion", representacion, "--salida", salida]

    # estas dos solo tienen sentido con palabras
    if representacion == "palabras":
        ngramas = input("N-gramas [1]: ").strip()
        if ngramas != "":
            args += ["--ngramas", ngramas]
        if input("¿Quitar palabras vacías? (s/n): ").strip().lower() == "s":
            args.append("--quitar-vacias")

    if input("¿TF sublineal? (s/n): ").strip().lower() == "s":
        args.append("--sublinear-tf")
    ejecutar(*args)


def elegir_carpeta(patron):
    """Enseña numeradas las carpetas que tienen un archivo que cumple el patrón y deja elegir una.

    Pre:  patron es relativo a la carpeta del proyecto, por ejemplo "resultados*/**/X.npz".
    Post: devuelve la carpeta elegida, o None si se elige 0 o no hay ninguna.
    """
    carpetas = []
    for x in sorted(RAIZ.glob(patron)):
        carpetas.append(x.parent.relative_to(RAIZ).as_posix())
    if not carpetas:
        print("No hay ninguna carpeta")
        return None

    for i, carpeta in enumerate(carpetas, 1):
        print(f"{i}) {carpeta}")
    print("0) Volver")
    opcion = pedir_numero(len(carpetas))
    if opcion == 0:
        return None
    return carpetas[opcion - 1]


def svd():
    """Deja elegir un TF-IDF ya hecho y lo reduce con SVD.

    Pre:  los TF-IDF están dentro de alguna carpeta resultados*/.
    Post: si se elige uno, matriz reducida guardada en <carpeta>_svd<dimensiones>.
    """
    print("\n¿Qué TF-IDF quieres reducir?")
    carpeta = elegir_carpeta("resultados*/**/X.npz")
    if carpeta is None:
        return

    dimensiones = input("Dimensiones [100]: ").strip()
    if dimensiones == "":
        dimensiones = "100"
    ejecutar("vectorizacion.svd", "--entrada", carpeta, "--dimensiones", dimensiones)


def densa(solo_reduccion):
    """Pregunta el conjunto y el modelo y lanza la vectorización densa.

    Pre:  los parámetros de PCA y UMAP están en vectorizacion_densa/config.json; para calcular
          embeddings hace falta el CSV del conjunto que indica ese config.
    Post: resultados guardados en resultados_roberta/<conjunto>/<modelo>/.
    """
    conjunto = elegir("¿Qué conjunto?", CONJUNTOS)
    if conjunto is None:
        return
    modelo = elegir("¿Qué modelo?", MODELOS_DENSOS)
    if modelo is None:
        return

    reduccion = json.loads(CONFIG_DENSA.read_text(encoding="utf-8"))["reduccion"]
    print(f"\nPCA + UMAP de vectorizacion_densa/config.json: varianza {reduccion['varianza']}, "
          f"min_dist {reduccion['min_dist']}")
    modulo = "vectorizacion_densa.reduccionDim" if solo_reduccion else "vectorizacion_densa"
    ejecutar(modulo, "--conjunto", conjunto, "--modelo", modelo)


def vectorizar():
    """Submenú de vectorización.

    Pre:  output/ tiene los CSV del preprocesador.
    Post: se ha lanzado la vectorización elegida, o nada si se elige 0.
    """
    print("\n--- Vectorizar ---")
    print("1) TF-IDF de palabras")
    print("2) TF-IDF de caracteres")
    print("3) SVD sobre un TF-IDF")
    print("4) RoBERTa: embeddings + PCA + UMAP")
    print("5) Solo PCA + UMAP sobre embeddings ya calculados")
    print("0) Volver")
    opcion = pedir_numero(5)

    if opcion == 1:
        tfidf("palabras")
    elif opcion == 2:
        tfidf("caracteres4")
    elif opcion == 3:
        svd()
    elif opcion == 4:
        densa(solo_reduccion=False)
    elif opcion == 5:
        densa(solo_reduccion=True)


def clustering():
    """Lanza el clustering con la entrada y la salida de config.json.

    Pre:  la "entrada" de config.json tiene X.npz o X.npy.
    Post: resultados del experimento o modelo.pkl guardados en la "salida" de config.json.
    """
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    print("\nEntrada:", config["entrada"])
    ejecutar("Clustering_jerarquico")


def banco_pruebas():
    """Compara todas las representaciones de un conjunto con experimentos.todas.

    Pre:  existen las carpetas del conjunto dentro de resultados_tfidf/.
    Post: rejilla.csv, resumen.json e informe.html en resultados_experimentos_todas_<conjunto>.
    """
    conjunto = elegir("¿Qué conjunto?", CONJUNTOS)
    if conjunto is None:
        return

    repeticiones = input("Repeticiones de estabilidad (0 = no medirla) [0]: ").strip()
    if repeticiones == "":
        repeticiones = "0"
    ejecutar("experimentos.todas", "--conjunto", conjunto, "--estabilidad", repeticiones)


def bootstrap():
    """Mide con bootstrap lo estable que es el clustering de una representación.

    Pre:  la carpeta elegida tiene X.npz o X.npy.
    Post: enseña el ARI y su intervalo para cada enlace con el k que se escribe.
    """
    print("\n¿Con qué representación?")
    carpeta = elegir_carpeta("resultados*/**/X.np[yz]")
    if carpeta is None:
        return

    k = input("k: ").strip()
    if k == "":
        return
    repeticiones = input("Repeticiones [50]: ").strip()
    if repeticiones == "":
        repeticiones = "50"
    ejecutar("experimentos.bootstrap", "--entrada", carpeta, "--k", k, "--repeticiones", repeticiones)


def figuras():
    """Genera las tablas y las figuras de la memoria.

    Pre:  existen las carpetas de resultados_tfidf/datos_1_a_600.
    Post: CSV y PNG guardados en la carpeta de salida.
    """
    salida = input("Carpeta de salida [resultados_memoria]: ").strip()
    if salida == "":
        salida = "resultados_memoria"
    ejecutar("experimentos.figuras_memoria", "--salida", salida)


def experimentar():
    """Submenú de experimentos.

    Pre:  las representaciones ya están hechas.
    Post: se ha lanzado el experimento elegido, o nada si se elige 0.
    """
    print("\n--- Experimentos ---")
    print("1) Banco de pruebas (todas las representaciones)")
    print("2) Bootstrap")
    print("3) Figuras de la memoria")
    print("0) Volver")
    opcion = pedir_numero(3)

    if opcion == 1:
        banco_pruebas()
    elif opcion == 2:
        bootstrap()
    elif opcion == 3:
        figuras()


def instancia_nueva():
    """Asigna un texto nuevo a un cluster con el modelo guardado.

    Pre:  existe el modelo de "mejor_modelo" y se entrenó con la "entrada" de config.json.
    Post: enseña el cluster y los usuarios más parecidos, o avisa si la entrada no sirve.
    """
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    entrada = RAIZ / config["entrada"]
    modelo = config["salida"] + "/" + config["mejor_modelo"]["archivo"]

    # para vectorizar el texto nuevo hace falta el modelo de TF-IDF o de SVD, o una carpeta de
    # RoBERTa (su config.json dice con qué modelo se embebió)
    config_densa = entrada / "config.json"
    densa = config_densa.exists() and "model_name" in json.loads(config_densa.read_text(encoding="utf-8"))
    if not densa and not (entrada / "vectorizador.joblib").exists() and not (entrada / "svd.joblib").exists():
        print("\nLa entrada de config.json no es TF-IDF, SVD ni RoBERTa: no se puede vectorizar el texto")
        return
    if not (RAIZ / modelo).exists():
        print(f"\nNo existe {modelo}. Genéralo en Clustering > modelo")
        return
    texto = input("Texto a consultar: ").strip()
    if texto == "":
        return
    ejecutar("experimentos.consultar", texto, "--entrada", config["entrada"], "--modelo", modelo)


def main():
    """Menú principal.

    Pre:  se ejecuta con el Python del proyecto (el del .venv).
    Post: repite el menú hasta que se elige 0.
    """
    while True:
        print("\n===== MENÚ =====")
        print("1) Preprocesar")
        print("2) Análisis de los datos")
        print("3) Vectorizar")
        print("4) Clustering")
        print("5) Experimentos")
        print("6) Instancia nueva")
        print("0) Salir")
        opcion = pedir_numero(6)

        if opcion == 1:
            preprocesar()
        elif opcion == 2:
            analisis()
        elif opcion == 3:
            vectorizar()
        elif opcion == 4:
            clustering()
        elif opcion == 5:
            experimentar()
        elif opcion == 6:
            instancia_nueva()
        else:
            break


if __name__ == "__main__":
    main()
