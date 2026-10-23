"""Embeddings densos por usuario y reducción PCA + UMAP, con los parámetros de config.json.

    python -m vectorizacion_densa [--conjunto datos_1_a_600] [--modelo roberta] [--sin-reduccion]

Guarda en resultados_roberta/<conjunto>/<modelo>/:
    sin_reduccion/                      un embedding por usuario, tal como sale del modelo
    PCA_umap_<varianza>_<min_dist>/     los mismos, reducidos con PCA + UMAP
Si sin_reduccion ya existe se reutiliza y solo se repite la reducción. Para recalcular los
embeddings, borra esa carpeta.
"""
import argparse
import json

from .reduccionDim import RAIZ, SIN_REDUCCION, cargar_config, carpeta_modelo, guardar, reducir

MODELOS = {
    "roberta": dict(
        model_name="FacebookAI/roberta-large",
        max_length=512, batch_size=32, normalizar=False,
    ),
    "all-roberta": dict(
        model_name="sentence-transformers/all-roberta-large-v1",
        max_length=256, batch_size=64, normalizar=True,
    ),
    "multilingue": dict(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        max_length=128, batch_size=128, normalizar=False,
    ),
}


def alinear(ids, X, n_posts, archivo_usuarios):
    """Deja los usuarios en el orden de la representación TF-IDF del mismo conjunto.

    Así el banco de pruebas puede comparar RoBERTa con TF-IDF: mismos usuarios y mismo orden.

    Pre:  ids, X y n_posts tienen una fila por usuario; archivo_usuarios es un usuarios.json de TF-IDF.
    Post: (ids, X, n_posts) filtrados y reordenados. Si el archivo no existe, se devuelven igual.
    """
    if not archivo_usuarios.is_file():
        print(f"No existe {archivo_usuarios}: se guardan todos los usuarios, en su orden.")
        return ids, X, n_posts
    referencia = [u["usuario"] for u in json.loads(archivo_usuarios.read_text(encoding="utf-8"))]
    posicion = {u: i for i, u in enumerate(ids)}
    sobran = sorted(set(ids) - set(referencia))
    faltan = [u for u in referencia if u not in posicion]
    if sobran:
        print(f"Se quitan {len(sobran)} usuarios que no están en TF-IDF: {', '.join(sobran)}")
    if faltan:
        print(f"Aviso: {len(faltan)} usuarios de TF-IDF no tienen embedding, el banco no podrá comparar.")
    filas = [posicion[u] for u in referencia if u in posicion]
    return ids[filas], X[filas], n_posts[filas]


def calcular_embeddings(conjunto: str, modelo: str, config: dict):
    """Calcula un embedding por usuario y lo guarda en sin_reduccion.

    Pre:  existe el CSV del conjunto indicado en config.json y modelo está en MODELOS.
    Post: carpeta sin_reduccion con X.npy, user_embeddings.npz, usuarios.json y config.json.
    """
    from .roberta import RobertaPorUsuario      # carga torch: solo cuando hace falta

    rutas = config["conjuntos"][conjunto]
    if not (RAIZ / rutas["datos"]).is_file():
        raise SystemExit(f"No existe {rutas['datos']}. Genera antes el CSV con el preprocesador.")
    vectorizador = RobertaPorUsuario(data_path=RAIZ / rutas["datos"], **MODELOS[modelo])
    post_df, user_df = vectorizador.fit_transform()
    print(f"Posts: {len(post_df)} | usuarios: {len(user_df)} | dimensiones: {vectorizador.user_embeddings.shape[1]}")

    ids, X, n_posts = alinear(user_df["subject_id"].to_numpy().astype(str), vectorizador.user_embeddings,
                              user_df["n_posts"].to_numpy(), RAIZ / rutas["usuarios"])
    guardar(carpeta_modelo(conjunto, modelo) / SIN_REDUCCION, X, ids, n_posts,
            {"conjunto": conjunto, "modelo": modelo, "reduccion": SIN_REDUCCION, "datos": rutas["datos"],
             **MODELOS[modelo]})


def main(argv=None):
    """
    Función principal encargada de coordinar la carga de configuración, la definición de argumentos de línea de comandos y el flujo de cálculo de embeddings densos y su posterior reducción dimensional.

    Precondiciones:
        - Existencia de un archivo de configuración válido y módulos requeridos disponibles en el entorno.

    Postcondiciones:
        - Se generan y guardan los embeddings densos y reducidos en el disco según los parámetros especificados.
    """
    config = cargar_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--conjunto", default=config["conjunto"], choices=list(config["conjuntos"]))
    parser.add_argument("--modelo", default=config["modelo"], choices=list(MODELOS))
    parser.add_argument("--sin-reduccion", action="store_true", help="Solo calcula los embeddings, sin PCA + UMAP.")
    args = parser.parse_args(argv)

    base = carpeta_modelo(args.conjunto, args.modelo) / SIN_REDUCCION
    if (base / "user_embeddings.npz").is_file():
        print(f"Ya existen los embeddings en {base}: se reutilizan.")
    else:
        calcular_embeddings(args.conjunto, args.modelo, config)
    if not args.sin_reduccion:
        reducir(args.conjunto, args.modelo, config["reduccion"])


if __name__ == "__main__":
    main()
