"""Vectoriza todos los usuarios del CSV: python -m vectorizacion --help."""

import argparse
import json
import hashlib
from pathlib import Path

import joblib
from scipy.sparse import save_npz
import sklearn

from .tfidf import crear_vectorizador, cargar_historiales_csv


def main(argv=None):
    """Ajusta TF-IDF con todos los usuarios útiles del CSV preprocesado.

    Requiere un CSV y una salida vacía. Exporta X.npz, usuarios.json, el modelo
    reutilizable y un resumen. El remuestreo corresponde al módulo de experimentos.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datos", type=Path, default=Path("output/data_preprocesado.csv"))
    parser.add_argument("--salida", type=Path,
                        help="Carpeta nueva o vacía; por defecto resultados_tfidf/datos_1_a_600/resultados_tfidf[_caracteres4]_completo.")
    parser.add_argument("--representacion", choices=("palabras", "caracteres4"), default="palabras")
    parser.add_argument("--ngramas", type=int, default=1, help="Longitud máxima para palabras; caracteres4 usa siempre 4.")
    parser.add_argument("--min-df", type=int, default=3)
    parser.add_argument("--max-df", type=float, default=1.0)
    parser.add_argument("--max-features", type=int)
    parser.add_argument("--sublinear-tf", action="store_true")
    parser.add_argument("--quitar-vacias", action="store_true",
                        help="Filtra las palabras vacias inglesas (NLTK), conservando las negaciones.")
    args = parser.parse_args(argv)
    if args.salida is None:
        args.salida = Path("resultados_tfidf", "datos_1_a_600",
                           "resultados_tfidf_caracteres4_completo" if args.representacion == "caracteres4"
                           else "resultados_tfidf_completo")
    if args.representacion == "caracteres4" and args.ngramas != 1:
        parser.error("No uses --ngramas con caracteres4: su longitud ya está fijada en 4.")
    if args.representacion == "caracteres4" and args.quitar_vacias:
        parser.error("--quitar-vacias no aplica a caracteres4: ahí no hay palabras que filtrar.")
    if args.salida.exists() and (not args.salida.is_dir() or any(args.salida.iterdir())):
        parser.error("La salida debe ser una carpeta nueva o vacía.")

    informe_entrada = {}
    try:
        usuarios = cargar_historiales_csv(args.datos, informe=informe_entrada)
    except (FileNotFoundError, ValueError) as e:
        parser.error(str(e))
    if len(usuarios) < args.min_df:
        parser.error("No hay suficientes usuarios para min_df.")

    vectorizador = crear_vectorizador(
        args.ngramas, args.min_df, args.max_df, args.max_features, args.sublinear_tf,
        representacion=args.representacion, quitar_vacias=args.quitar_vacias,
    )
    matriz = vectorizador.fit_transform([u.publicaciones for u in usuarios])
    args.salida.mkdir(parents=True, exist_ok=True)
    save_npz(args.salida / "X.npz", matriz)
    joblib.dump(vectorizador, args.salida / "vectorizador.joblib")

    resumen = {
        "modo": "completo",
        "sklearn_version": sklearn.__version__,
        "configuracion": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "usuarios": len(usuarios),
        "entrada": informe_entrada,
        "datos_sha256": hashlib.sha256(args.datos.read_bytes()).hexdigest(),
        "publicaciones_totales": sum(u.publicaciones_totales for u in usuarios),
        "publicaciones_utiles": sum(len(u.publicaciones) for u in usuarios),
        "dimensiones": len(vectorizador.get_feature_names_out()),
        "analizador": vectorizador.analyzer,
        "ngram_range": list(vectorizador.ngram_range),
        "matriz": {"archivo": "X.npz", "shape": list(matriz.shape), "nnz": matriz.nnz},
        "usuarios_vector_cero": [
            u.usuario for u, nnz in zip(usuarios, matriz.getnnz(axis=1)) if nnz == 0
        ],
        "alcance": "TF-IDF ajustado con todos los usuarios del CSV. El remuestreo de X evalúa el clustering con representación fija.",
    }
    filas = [
        {"fila": i, "usuario": u.usuario, "archivo": u.archivo,
         "publicaciones_utiles": len(u.publicaciones)}
        for i, u in enumerate(usuarios)
    ]
    (args.salida / "usuarios.json").write_text(
        json.dumps(filas, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.salida / "resumen.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(resumen, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
