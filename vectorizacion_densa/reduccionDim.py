"""Reducción PCA + UMAP de los embeddings de usuario, con los parámetros de config.json.

    python -m vectorizacion_densa.reduccionDim [--conjunto datos_1_a_600] [--modelo roberta]

Lee resultados_roberta/<conjunto>/<modelo>/sin_reduccion y guarda en
resultados_roberta/<conjunto>/<modelo>/PCA_umap_<varianza>_<min_dist>.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parent.parent
CONFIG = Path(__file__).resolve().parent / "config.json"
RESULTADOS = RAIZ / "resultados_roberta"
SIN_REDUCCION = "sin_reduccion"


def cargar_config(ruta=CONFIG) -> dict:
    """Lee config.json y comprueba los parámetros de la reducción.

    Pre:  ruta es un JSON con conjunto, modelo, reduccion y conjuntos.
    Post: el diccionario leído. Lanza ValueError si la varianza no está entre 0 y 1
          o min_dist no está en [0, 1].
    """
    config = json.loads(Path(ruta).read_text(encoding="utf-8"))
    reduccion = config["reduccion"]
    if not 0 < reduccion["varianza"] < 1:
        raise ValueError("La varianza de PCA tiene que estar entre 0 y 1 (por ejemplo 0.9).")
    if not 0 <= reduccion["min_dist"] <= 1:
        raise ValueError("El min_dist de UMAP tiene que estar entre 0 y 1.")
    return config


def nombre_reduccion(reduccion: dict) -> str:
    """Pre: reduccion tiene varianza y min_dist.  Post: nombre de la carpeta, por ejemplo PCA_umap_0.9_0.0."""
    return f"PCA_umap_{float(reduccion['varianza'])}_{float(reduccion['min_dist'])}"


def carpeta_modelo(conjunto: str, modelo: str) -> Path:
    """Pre: conjunto y modelo son nombres de carpeta.  Post: resultados_roberta/<conjunto>/<modelo>."""
    return RESULTADOS / conjunto / modelo


def construir_reductor(n_usuarios, varianza=0.90, componentes=15, vecinos=30, min_dist=0.0, semilla=42):
    """Escalado -> PCA -> UMAP empaquetado en un Pipeline de scikit-learn.

    Pre:  n_usuarios >= 2 y 0 < varianza < 1.
    Post: Pipeline sin ajustar; fit_transform devuelve una matriz (n_usuarios, componentes).
    """
    return Pipeline([
        ("escalado", StandardScaler()),
        ("pca", PCA(n_components=varianza, svd_solver="full")),
        ("umap", umap.UMAP(
            n_components=componentes,
            n_neighbors=min(vecinos, n_usuarios - 1),
            min_dist=min_dist,
            metric="cosine",
            random_state=semilla,
        )),
    ])


def guardar(carpeta: Path, X, ids, n_posts, config_usada: dict):
    """Guarda una representación con el mismo formato que las de TF-IDF.

    Pre:  X tiene una fila por usuario de ids y n_posts, en el mismo orden.
    Post: carpeta con X.npy, user_embeddings.npz, usuarios.json (con la fila de cada usuario)
          y config.json con los parámetros usados.
    """
    carpeta.mkdir(parents=True, exist_ok=True)
    np.save(carpeta / "X.npy", np.asarray(X, dtype=np.float32), allow_pickle=False)
    np.savez(carpeta / "user_embeddings.npz", embeddings=X, subject_id=np.asarray(ids, dtype=str), n_posts=n_posts)
    usuarios = [{"fila": i, "usuario": str(u), "n_posts": int(n)} for i, (u, n) in enumerate(zip(ids, n_posts))]
    (carpeta / "usuarios.json").write_text(json.dumps(usuarios, ensure_ascii=False, indent=2), encoding="utf-8")
    (carpeta / "config.json").write_text(json.dumps(config_usada, ensure_ascii=False, indent=2), encoding="utf-8")


def reducir(conjunto: str, modelo: str, reduccion: dict) -> Path:
    """Aplica PCA + UMAP a los embeddings sin reducir de un conjunto y un modelo.

    Pre:  existe resultados_roberta/<conjunto>/<modelo>/sin_reduccion/user_embeddings.npz
          y reduccion tiene varianza, min_dist, componentes, vecinos y semilla.
    Post: carpeta PCA_umap_<varianza>_<min_dist> con la matriz reducida y reductor.joblib
          (para transformar usuarios nuevos). Devuelve su ruta.
    """
    base = carpeta_modelo(conjunto, modelo) / SIN_REDUCCION
    if not (base / "user_embeddings.npz").is_file():
        raise FileNotFoundError(f"No existe {base}. Calcula antes los embeddings: "
                                f"python -m vectorizacion_densa --conjunto {conjunto} --modelo {modelo}")
    datos = np.load(base / "user_embeddings.npz")
    X = datos["embeddings"]

    reductor = construir_reductor(len(X), reduccion["varianza"], reduccion["componentes"],
                                  reduccion["vecinos"], reduccion["min_dist"], reduccion["semilla"])
    X_reducida = reductor.fit_transform(X)
    pca = reductor.named_steps["pca"]
    print(f"PCA: {pca.n_components_} componentes ({pca.explained_variance_ratio_.sum():.1%} de la varianza)")

    salida = carpeta_modelo(conjunto, modelo) / nombre_reduccion(reduccion)
    config_base = json.loads((base / "config.json").read_text(encoding="utf-8"))
    guardar(salida, X_reducida, datos["subject_id"], datos["n_posts"],
            {**config_base, "reduccion": nombre_reduccion(reduccion), "parametros_reduccion": reduccion})
    joblib.dump(reductor, salida / "reductor.joblib")
    print(f"Matriz reducida: {X_reducida.shape}\nGuardado en: {salida}")
    return salida


def main(argv=None):
    config = cargar_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--conjunto", default=config["conjunto"], choices=list(config["conjuntos"]))
    parser.add_argument("--modelo", default=config["modelo"], help="Carpeta del modelo: roberta, all-roberta, multilingue.")
    args = parser.parse_args(argv)
    try:
        reducir(args.conjunto, args.modelo, config["reduccion"])
    except FileNotFoundError as error:
        raise SystemExit(str(error))


if __name__ == "__main__":
    main()
