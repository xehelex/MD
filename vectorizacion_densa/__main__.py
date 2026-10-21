import argparse
import json
from pathlib import Path
import numpy as np
import sklearn
import umap
import joblib

from .roberta import RobertaPorUsuario
from .reduccionDim import construir_reductor

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


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Pipeline de extracción, agregación y reducción de embeddings de usuarios.")
    parser.add_argument("--datos", type=Path, default=Path("output/data.csv"),
                        help="Ruta al archivo CSV con los datos de entrada.")
    parser.add_argument("--modelo", type=str, default="roberta",
                        choices=list(MODELOS.keys()),
                        help="Modelo a utilizar: roberta, all-roberta, multilingue")
    parser.add_argument("--reduccion", type=str, default="sin_reduccion",
                        help="Tipo de reducción: 'sin_reduccion' o formato 'PCA_umap_0.9_<min_dist>' (ej. PCA_umap_0.9_0.0)")
    args = parser.parse_args(argv)

    if args.modelo not in MODELOS:
        raise SystemExit(f"Modelo desconocido '{args.modelo}'. Opciones: {list(MODELOS)}")

    config = MODELOS[args.modelo]

    # Directorios base
    modelo_dir = Path("resultados_roberta") / args.modelo
    output_dir = modelo_dir / args.reduccion
    output_dir.mkdir(parents=True, exist_ok=True)

    sin_reduccion_dir = modelo_dir / "sin_reduccion"
    base_user_npz = sin_reduccion_dir / "user_embeddings.npz"

    # REUTILIZACIÓN INTELIGENTE: Si NO estamos en sin_reduccion y ya existe la base, la cargamos sin tocar el modelo
    if args.reduccion != "sin_reduccion" and base_user_npz.exists():
        print(f"-> Encontrados embeddings base de usuarios en '{sin_reduccion_dir}'. Reutilizando caché...")
        data_base = np.load(base_user_npz)
        X = data_base["embeddings"]
        subject_ids = data_base["subject_id"]
        n_posts = data_base["n_posts"]

        print(f"Número de usuarios cargados desde caché: {len(subject_ids)}")
        print(f"Shape original de X: {X.shape}")

    else:
        # Si es sin_reduccion o no existe la base, ejecutamos el pipeline completo con RoBERTa
        if args.reduccion != "sin_reduccion":
            print(f"-> No se encontró caché en '{sin_reduccion_dir}'. Ejecutando extracción completa con el modelo...")

        vectorizer = RobertaPorUsuario(data_path=args.datos, **config)
        post_df, user_df = vectorizer.fit_transform()

        X = vectorizer.user_embeddings
        subject_ids = user_df["subject_id"].to_numpy().astype(str)
        n_posts = user_df["n_posts"].to_numpy()

        print("Número de posts procesados:", len(post_df))
        print("Número de usuarios:", len(user_df))
        print("Shape original de X:", X.shape)

    # Aplicar reducción si corresponde
    if args.reduccion == "sin_reduccion":
        X_final = X
        print("Pipeline sin reducción aplicada.")
    elif args.reduccion.startswith("PCA_umap_"):
        try:
            min_dist = float(args.reduccion.split("_")[-1])
        except ValueError:
            min_dist = 0.0

        print(f"Aplicando reducción PCA + UMAP con min_dist={min_dist}...")

        reductor = construir_reductor(len(X), min_dist=min_dist)
        X_final = reductor.fit_transform(X).astype(np.float32)

        pca = reductor.named_steps["pca"]
        print(f"PCA: {pca.n_components_} componentes ({pca.explained_variance_ratio_.sum():.1%} de la varianza)")
        print("Shape reducida de X:", X_final.shape)

        # Guardar artefactos de la reducción
        joblib.dump(reductor, output_dir / "reductor.joblib")
    else:
        raise SystemExit(
            f"Tipo de reducción desconocida: '{args.reduccion}'. Usa 'sin_reduccion' o formato 'PCA_umap_...'.")

    # Guardado de ficheros esenciales de usuarios (sin rastro de post_embeddings)
    np.save(output_dir / "X.npy", X_final)

    np.savez(
        output_dir / "user_embeddings.npz",
        embeddings=X_final,
        subject_id=subject_ids,
        n_posts=n_posts,
    )

    usuarios_data = [{"usuario": str(u), "n_posts": int(n)} for u, n in zip(subject_ids, n_posts)]
    (output_dir / "usuarios.json").write_text(
        json.dumps(usuarios_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    config_data = {
        "modelo": args.modelo,
        "reduccion": args.reduccion,
        "datos": str(args.datos),
        **config
    }
    (output_dir / "config.json").write_text(
        json.dumps(config_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("Guardado correctamente en:", output_dir.resolve())


if __name__ == "__main__":
    main()