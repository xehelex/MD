"""Reduce TF-IDF con TruncatedSVD y normaliza L2: python -m vectorizacion.svd.

Precondiciones:
    - --entrada es una carpeta TF-IDF con X.npz dispersa y un usuarios.json válido.
      Si existen vectorizador.joblib y resumen.json, solo se usan para las huellas.
    - 1 <= --dimensiones <= min(n_usuarios, n_columnas) - 1.
    - 0 <= --semilla < 2**32.
    - --salida no existe o está vacía (por defecto <entrada>_svd<dimensiones>,
      junto a la entrada).
    - La matriz tiene varianza suficiente para que SVD dé estadísticas finitas.

Postcondiciones:
    - Se crea --salida con:
        X.npy          float32 de forma (n_usuarios, dimensiones), filas
                       normalizadas L2 y en el mismo orden que la entrada. Las filas
                       cero se conservan y se listan en el resumen.
        svd.joblib     Pipeline(TruncatedSVD aleatorizado con n_iter=7 + Normalizer L2)
                       ajustado con todos los usuarios; transforma TF-IDF nuevos del
                       mismo vectorizador.
        usuarios.json  copia byte a byte de la entrada.
        resumen.json   configuración, SHA-256 del origen y varianza explicada
                       (medida antes de L2).
    - Con la misma entrada, semilla y versión de scikit-learn, el resultado es reproducible.
    - El TF-IDF de entrada no se modifica.
"""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from .matrices import cargar_matriz, cargar_usuarios


def huella(archivo):
    """SHA-256 de un fichero, leído por bloques de 1 MB.

    Pre:  archivo existe y se puede leer.
    Post: el hash en hexadecimal.
    """
    digest = hashlib.sha256()
    with Path(archivo).open("rb") as f:
        for bloque in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def reducir(entrada, salida, dimensiones=100, semilla=42):
    """Ajusta sobre todos los usuarios; conserva las filas y nunca sobrescribe.

    Pre:  entrada tiene X.npz y usuarios.json válidos, salida no existe o está vacía,
          1 <= dimensiones <= min(X.shape) - 1 y 0 <= semilla < 2**32.
    Post: en salida quedan X.npy (float32, filas con norma L2), svd.joblib, usuarios.json y
          resumen.json; devuelve el resumen. Si no se cumple la precondición, ValueError.
    """
    entrada, salida = Path(entrada), Path(salida)
    if salida.exists() and (not salida.is_dir() or any(salida.iterdir())):
        raise ValueError("La salida debe ser una carpeta nueva o vacía.")
    X = cargar_matriz(entrada)
    if not sparse.issparse(X):
        raise ValueError("La entrada de SVD debe ser el TF-IDF disperso X.npz.")
    usuarios = cargar_usuarios(entrada, X.shape[0])
    limite = min(X.shape) - 1
    if not 1 <= dimensiones <= limite:
        raise ValueError(f"dimensiones debe estar entre 1 y {limite} para una matriz {X.shape}.")
    if not 0 <= semilla < 2**32:
        raise ValueError("semilla debe estar entre 0 y 2**32 - 1.")

    # No se densifica el TF-IDF: solo la proyección tiene formato denso.
    modelo = make_pipeline(
        TruncatedSVD(n_components=dimensiones, algorithm="randomized", n_iter=7, random_state=semilla),
        Normalizer(norm="l2"),
    )
    reducida = modelo.fit_transform(X).astype(np.float32, copy=False)
    svd = modelo.named_steps["truncatedsvd"]
    proporciones = svd.explained_variance_ratio_
    if not np.isfinite(reducida).all() or not np.isfinite(proporciones).all():
        raise ValueError("SVD no produce estadísticas finitas; comprueba que la entrada tenga varianza.")
    if reducida.shape != (X.shape[0], dimensiones):
        raise ValueError("SVD no produjo el número de dimensiones solicitado.")

    resumen = {
        "modo": "completo",
        "representacion": "tfidf_svd",
        "sklearn_version": sklearn.__version__,
        "configuracion": {"entrada": str(entrada), "salida": str(salida), "dimensiones": dimensiones,
                          "semilla": semilla, "algoritmo": "randomized", "n_iter": 7, "normalizacion": "l2"},
        "origen": {"shape": list(X.shape), "archivos_sha256": {
            nombre: huella(entrada / nombre)
            for nombre in ("X.npz", "usuarios.json", "vectorizador.joblib", "resumen.json")
            if (entrada / nombre).is_file()
        }},
        "usuarios": X.shape[0],
        "dimensiones": dimensiones,
        "matriz": {"archivo": "X.npy", "shape": list(reducida.shape), "dtype": str(reducida.dtype)},
        "varianza_explicada_acumulada": float(proporciones.sum()),
        "varianza_explicada_por_componente": proporciones.tolist(),
        "nota_varianza": "Proporción de varianza calculada por SVD antes de normalizar L2; no mide calidad del clustering.",
        "usuarios_vector_cero": [u["usuario"] for u, fila in zip(usuarios, reducida) if not np.any(fila)],
        "alcance": "SVD ajustado con todos los usuarios, sobre TF-IDF previamente ajustado. El remuestreo de X evalúa clustering con representación fija.",
    }
    salida.mkdir(parents=True, exist_ok=True)
    np.save(salida / "X.npy", reducida, allow_pickle=False)
    joblib.dump(modelo, salida / "svd.joblib", compress=3)
    # Conserva exactamente los metadatos y el orden de las filas originales.
    (salida / "usuarios.json").write_bytes((entrada / "usuarios.json").read_bytes())
    (salida / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return resumen


def main(argv=None):
    """Punto de entrada: python -m vectorizacion.svd --entrada ... [--dimensiones N].

    Pre:  argumentos de la línea de comandos (ver --help).
    Post: como reducir, con salida <entrada>_svd<dimensiones> por defecto; imprime el resumen
          o termina con el error de argparse.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--entrada", type=Path, required=True, help="Carpeta TF-IDF con X.npz y usuarios.json.")
    parser.add_argument("--salida", type=Path, help="Carpeta nueva o vacía; por defecto <entrada>_svd<dimensiones>.")
    parser.add_argument("--dimensiones", type=int, default=100)
    parser.add_argument("--semilla", type=int, default=42)
    args = parser.parse_args(argv)
    salida = args.salida if args.salida is not None else args.entrada.with_name(f"{args.entrada.name}_svd{args.dimensiones}")
    try:
        resumen = reducir(args.entrada, salida, args.dimensiones, args.semilla)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Matriz densa: {resumen['matriz']['shape']}; normalización L2.")
    print(f"Varianza explicada acumulada: {resumen['varianza_explicada_acumulada']:.2%} (antes de L2).")
    print(f"Usuarios con vector cero: {len(resumen['usuarios_vector_cero'])}.")
    print(f"Resultados en: {salida.resolve()}")


if __name__ == "__main__":
    main()
