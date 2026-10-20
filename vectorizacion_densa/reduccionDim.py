import json
from pathlib import Path

import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Para cargar y guardar
CARPETA = Path(__file__).resolve().parent.parent / "resultados_roberta"
SALIDA = Path(__file__).resolve().parent.parent / "resultados_roberta_PCA_umap"


def reducir_pca(X, var_pca=0.90):
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=var_pca, svd_solver="full")
    X = pca.fit_transform(X)
    print(f"PCA: {pca.n_components_} componentes "
          f"({pca.explained_variance_ratio_.sum():.1%} de la varianza)")
    return X


def reducir_umap(X, n_umap=15, min_dist=0.0, n_neighbors=30, seed=42):
    return umap.UMAP(
        n_components=n_umap,
        n_neighbors=min(n_neighbors, len(X) - 1),
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
    ).fit_transform(X)


def main():
    print("Buscando en:", CARPETA)

    data = np.load(CARPETA / "user_embeddings.npz")
    X = data["embeddings"]
    ids = data["subject_id"]
    n_posts = data["n_posts"]

    X_pca = reducir_pca(X)
    X_clu = reducir_umap(X_pca, n_umap=15, min_dist=0.0)

    SALIDA.mkdir(parents=True, exist_ok=True)
    np.save(SALIDA / "X.npy", X_clu.astype(np.float32), allow_pickle=False)

    # Mismo orden de filas que X.npy: la fila i de X es el usuario i de este json
    usuarios = [{"usuario": str(u), "n_posts": int(n)} for u, n in zip(ids, n_posts)]
    (SALIDA / "usuarios.json").write_text(
        json.dumps(usuarios, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Matriz final:", X_clu.shape)
    print("Guardado en:", SALIDA)


if __name__ == "__main__":
    main()