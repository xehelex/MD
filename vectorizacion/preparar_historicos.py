"""Une y alinea TF-IDF históricos sin reajustar sus valores ni su vocabulario.

Precondiciones:
    - --entrada es una carpeta histórica con X_train.npz, X_test.npz,
      usuarios_train.json, usuarios_test.json, vectorizador.joblib y resumen.json.
    - Train y test tienen el mismo número de columnas, matrices finitas y no vacías,
      y no comparten IDs.
    - --referencia es una representación válida (X + usuarios.json) y todos sus
      usuarios están en el histórico.
    - --salida no existe o está vacía.

Postcondiciones:
    - Se crea --salida con:
        X.npz               train y test unidos y reordenados con los mismos usuarios
                            y el mismo orden que la referencia; valores sin recalcular.
        vectorizador.joblib copia exacta del histórico (ajustado solo con train).
        usuarios.json       fila nueva, más particion_original y fila_original.
        resumen.json        SHA-256 de los archivos de origen, usuarios excluidos
                            (los que no están en la referencia) y vectores cero.
    - Los archivos de --entrada y de --referencia no se modifican.
"""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from scipy import sparse

from .matrices import cargar_matriz, cargar_usuarios
from .svd import huella


def leer_particion(entrada, particion):
    """Lee la matriz y los usuarios de una partición histórica (train o test).

    Pre:  entrada (Path) tiene X_<particion>.npz y usuarios_<particion>.json.
    Post: (X en CSR, usuarios) comprobados: matriz real, finita y no vacía y un usuario por
          fila con fila == posición. ValueError si no.
    """
    X = sparse.load_npz(entrada / f"X_{particion}.npz").tocsr()
    usuarios = json.loads((entrada / f"usuarios_{particion}.json").read_text(encoding="utf-8"))
    if X.ndim != 2 or min(X.shape) < 1 or X.dtype.kind not in "fiu" or not np.isfinite(X.data).all():
        raise ValueError(f"Matriz {particion} inválida: debe ser real, finita y no vacía.")
    if not isinstance(usuarios, list) or len(usuarios) != X.shape[0]:
        raise ValueError(f"Debe haber un usuario por fila en {particion}.")
    for i, u in enumerate(usuarios):
        if (not isinstance(u, dict) or u.get("fila") != i
                or not isinstance(u.get("usuario"), str) or not u["usuario"].strip()):
            raise ValueError(f"Fila o ID inválido en {particion}, posición {i}.")
    return X, usuarios


def preparar(entrada, referencia, salida):
    """Junta train y test y los reordena con los usuarios y el orden de la referencia.

    Pre:  entrada es una carpeta histórica (X_train/X_test, usuarios_*, vectorizador.joblib y
          resumen.json), referencia tiene X y usuarios.json, y salida no existe o está vacía.
    Post: en salida quedan X.npz alineado, vectorizador.joblib, usuarios.json y resumen.json;
          devuelve el resumen. ValueError si faltan usuarios o hay IDs duplicados.
    """
    entrada, referencia, salida = Path(entrada), Path(referencia), Path(salida)
    if salida.exists() and (not salida.is_dir() or any(salida.iterdir())):
        raise ValueError("La salida debe ser una carpeta nueva o vacía.")
    matrices, usuarios = [], []
    for particion in ("train", "test"):
        X, filas = leer_particion(entrada, particion)
        matrices.append(X)
        usuarios.extend({**u, "particion_original": particion, "fila_original": u["fila"]} for u in filas)
    if matrices[0].shape[1] != matrices[1].shape[1]:
        raise ValueError("Train y test deben tener el mismo número de columnas.")
    por_id = {u["usuario"]: i for i, u in enumerate(usuarios)}
    if len(por_id) != len(usuarios):
        raise ValueError("Hay IDs duplicados dentro de las particiones o entre train y test.")
    X_referencia = cargar_matriz(referencia)
    ids = [u["usuario"] for u in cargar_usuarios(referencia, X_referencia.shape[0])]
    faltantes = sorted(set(ids) - set(por_id))
    if faltantes:
        raise ValueError(f"Faltan usuarios de la referencia en las matrices antiguas: {faltantes}")
    indices = [por_id[usuario] for usuario in ids]
    alineada = sparse.vstack(matrices, format="csr")[indices].tocsr()
    filas = [{**usuarios[indice], "fila": fila} for fila, indice in enumerate(indices)]
    archivos = ["X_train.npz", "X_test.npz", "usuarios_train.json", "usuarios_test.json",
                "vectorizador.joblib", "resumen.json"]
    hashes = {nombre: huella(entrada / nombre) for nombre in archivos}
    resumen_original = json.loads((entrada / "resumen.json").read_text(encoding="utf-8"))
    resumen = {
        "modo": "historico_alineado",
        "configuracion": {"entrada": str(entrada), "referencia": str(referencia), "salida": str(salida)},
        "usuarios": len(ids), "dimensiones": alineada.shape[1],
        "matriz": {"archivo": "X.npz", "shape": list(alineada.shape), "nnz": alineada.nnz},
        "origen": {"shape_train": list(matrices[0].shape), "shape_test": list(matrices[1].shape),
                   "archivos_sha256": hashes, "resumen_original": resumen_original},
        "usuarios_referencia_sha256": huella(referencia / "usuarios.json"),
        "usuarios_excluidos": sorted(set(por_id) - set(ids)),
        "usuarios_vector_cero": [u for u, nnz in zip(ids, alineada.getnnz(axis=1)) if nnz == 0],
        "ajuste_tfidf": "Se conserva el modelo histórico ajustado solo con train; no se recalculan vocabulario, IDF, TF ni normalización.",
        "usuarios_ajuste_original": matrices[0].shape[0],
        "alcance": "Mismos usuarios y orden que la referencia. El preprocesamiento y la población de ajuste difieren de las representaciones nuevas; esta comparación evalúa las variantes completas.",
    }
    salida.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(salida / "X.npz", alineada)
    shutil.copy2(entrada / "vectorizador.joblib", salida / "vectorizador.joblib")
    (salida / "usuarios.json").write_text(json.dumps(filas, ensure_ascii=False, indent=2), encoding="utf-8")
    (salida / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return resumen


def main(argv=None):
    """Punto de entrada: python -m vectorizacion.preparar_historicos --entrada ...

    Pre:  argumentos de la línea de comandos (ver --help).
    Post: como preparar, con salida <entrada>_historico_alineado por defecto; si algo falla,
          termina con el error de argparse.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--entrada", type=Path, required=True, help="Carpeta histórica con train y test.")
    parser.add_argument("--referencia", type=Path, default=Path("resultados_tfidf/datos_1_a_600/resultados_tfidf_completo"))
    parser.add_argument("--salida", type=Path, help="Por defecto <entrada>_historico_alineado.")
    args = parser.parse_args(argv)
    salida = args.salida if args.salida is not None else args.entrada.with_name(args.entrada.name + "_historico_alineado")
    try:
        resumen = preparar(args.entrada, args.referencia, salida)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Matriz alineada: {resumen['matriz']['shape']}; usuarios excluidos: {resumen['usuarios_excluidos']}.")
    print(f"{resumen['ajuste_tfidf']}")
    print(f"Resultados en: {salida.resolve()}")


if __name__ == "__main__":
    main()
