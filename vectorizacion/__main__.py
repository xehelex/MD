"""Ejecutar desde la carpeta de datos: python -m vectorizacion --help."""

import argparse
import json
import math
from pathlib import Path
import random

import joblib
from scipy.sparse import save_npz
import sklearn

from .tfidf import cargar_historiales, crear_vectorizador


def main():
    """Ajusta TF-IDF en train y exporta matrices y modelo para datos nuevos.

    Requiere una carpeta XML y una salida vacía. La partición se hace por usuario
    y solo entrenamiento participa en el aprendizaje del vocabulario y del IDF.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datos", type=Path, default=Path("."))
    parser.add_argument("--salida", type=Path, default=Path("resultados_tfidf"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--representacion", choices=("palabras", "caracteres4"), default="palabras")
    parser.add_argument("--ngramas", type=int, default=1, help="Longitud máxima para palabras; caracteres4 usa siempre 4.")
    parser.add_argument("--min-df", type=int, default=3)
    parser.add_argument("--max-df", type=float, default=1.0)
    parser.add_argument("--max-features", type=int)
    parser.add_argument("--sublinear-tf", action="store_true")
    parser.add_argument("--quitar-vacias", action="store_true",
                        help="Filtra las palabras vacias inglesas (NLTK), conservando las negaciones.")
    args = parser.parse_args()
    if not 0 < args.test_size < 1:
        parser.error("--test-size debe estar entre 0 y 1.")
    if args.representacion == "caracteres4" and args.ngramas != 1:
        parser.error("No uses --ngramas con caracteres4: su longitud ya está fijada en 4.")
    if args.representacion == "caracteres4" and args.quitar_vacias:
        parser.error("--quitar-vacias no aplica a caracteres4: ahí no hay palabras que filtrar.")
    if args.salida.exists() and (not args.salida.is_dir() or any(args.salida.iterdir())):
        parser.error("La salida debe ser una carpeta nueva o vacía.")

    usuarios = cargar_historiales(args.datos)
    indices = list(range(len(usuarios)))
    random.Random(args.seed).shuffle(indices)
    n_test = math.ceil(len(indices) * args.test_size)
    train = [usuarios[i] for i in sorted(indices[n_test:])]
    test = [usuarios[i] for i in sorted(indices[:n_test])]
    if len(train) < args.min_df:
        parser.error("No hay suficientes usuarios de entrenamiento para min_df.")

    vectorizador = crear_vectorizador(
        args.ngramas, args.min_df, args.max_df, args.max_features, args.sublinear_tf,
        representacion=args.representacion, quitar_vacias=args.quitar_vacias,
    )
    x_train = vectorizador.fit_transform([u.publicaciones for u in train])
    x_test = vectorizador.transform([u.publicaciones for u in test])
    args.salida.mkdir(parents=True, exist_ok=True)
    save_npz(args.salida / "X_train.npz", x_train)
    save_npz(args.salida / "X_test.npz", x_test)
    joblib.dump(vectorizador, args.salida / "vectorizador.joblib")

    resumen = {
        "sklearn_version": sklearn.__version__,
        "configuracion": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "usuarios": len(usuarios),
        "publicaciones_totales": sum(u.publicaciones_totales for u in usuarios),
        "publicaciones_utiles": sum(len(u.publicaciones) for u in usuarios),
        "dimensiones": len(vectorizador.get_feature_names_out()),
        "analizador": vectorizador.analyzer,
        "ngram_range": list(vectorizador.ngram_range),
        "train": {"shape": list(x_train.shape), "nnz": x_train.nnz},
        "test": {"shape": list(x_test.shape), "nnz": x_test.nnz},
    }
    for nombre, grupo, matriz in (("train", train, x_train), ("test", test, x_test)):
        resumen[nombre]["usuarios_vector_cero"] = [
            u.usuario for u, nnz in zip(grupo, matriz.getnnz(axis=1)) if nnz == 0
        ]
        filas = [
            {"fila": i, "usuario": u.usuario, "archivo": u.archivo,
             "publicaciones_utiles": len(u.publicaciones)}
            for i, u in enumerate(grupo)
        ]
        (args.salida / f"usuarios_{nombre}.json").write_text(
            json.dumps(filas, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (args.salida / "resumen.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(resumen, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
