import numpy as np
from pathlib import Path

from .roberta import RobertaPorUsuario

def main(argv=None):
    output_dir = Path("resultados_roberta")
    output_dir.mkdir(parents=True, exist_ok=True)

    vectorizer = RobertaPorUsuario(
        data_path="output/data.csv",
        batch_size=32,  # lotes, según lo que aguante la GPU
        max_length=512,
    )

    post_df, user_df = vectorizer.fit_transform()

    # Matriz para clustering (usuarios*dimEmbeddings)
    X = vectorizer.user_embeddings

    print("Número de posts:", len(post_df))
    print("Número de usuarios:", len(user_df))
    print("Shape de X:", X.shape)

    # Embeddings por usuario (misma fila que user_ids.csv)
    np.save("resultados_roberta/user_embeddings.npy", X)
    user_df.to_csv("resultados_roberta/user_ids.csv", index=False)

    # Embeddings por post (para posibles futuras agregaciones extra)
    np.save("resultados_roberta/post_embeddings.npy", vectorizer.post_embeddings)
    post_df[["subject_id"]].to_csv("resultados_roberta/post_ids.csv", index=False)

    # Guardar: embeddings e ids en el mismo archivo
    np.savez(
        "resultados_roberta/user_embeddings.npz",
        embeddings=X,
        subject_id=user_df["subject_id"].to_numpy().astype(str),
        n_posts=user_df["n_posts"].to_numpy(),
    )

    # Cargar
    data = np.load("resultados_roberta/user_embeddings.npz")
    X = data["embeddings"]
    ids = data["subject_id"]

if __name__ == "__main__":
    main()