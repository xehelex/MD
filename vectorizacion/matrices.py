"""Carga común de representaciones dispersas y densas y de sus usuarios.

Precondiciones:
    - cargar_matriz: la carpeta contiene exactamente uno de X.npz (scipy) o X.npy
      (numpy, sin pickle).
    - cargar_usuarios: la carpeta contiene usuarios.json, una lista con n_filas
      diccionarios donde la posición i tiene fila == i y un "usuario" de texto no vacío.

Postcondiciones:
    - cargar_matriz devuelve la matriz sin cambiar su formato: CSR si viene de
      X.npz, ndarray si viene de X.npy. Garantiza que es 2D, numérica, no vacía y finita.
    - cargar_usuarios devuelve la lista tal cual, con los IDs comprobados como
      únicos y alineados con las filas.
    - Si algo no se cumple: FileNotFoundError si falta la matriz, ValueError en el
      resto de casos. Nunca se escribe en disco.
"""

import json
from pathlib import Path

import numpy as np
from scipy import sparse


def cargar_matriz(carpeta):
    """Lee exactamente una X.npz (CSR) o X.npy (densa), sin convertir su formato.

    Pre:  carpeta contiene exactamente uno de X.npz o X.npy.
    Post: la matriz sin cambiar su formato (CSR o ndarray), 2D, numérica, no vacía y finita.
          FileNotFoundError si no hay ninguna; ValueError en el resto de casos.
    """
    carpeta = Path(carpeta)
    archivos = [p for p in (carpeta / "X.npz", carpeta / "X.npy") if p.is_file()]
    if not archivos:
        raise FileNotFoundError(f"No existe X.npz ni X.npy en {carpeta}. Genera la representación completa.")
    if len(archivos) != 1:
        raise ValueError(f"Hay X.npz y X.npy en {carpeta}; usa una carpeta por representación.")
    archivo = archivos[0]
    X = sparse.load_npz(archivo).tocsr() if archivo.suffix == ".npz" else np.load(archivo, allow_pickle=False)
    if X.ndim != 2 or min(X.shape) < 1 or X.dtype.kind not in "fiu":
        raise ValueError(f"La matriz de {carpeta} debe ser bidimensional, real y no vacía.")
    valores = X.data if sparse.issparse(X) else X
    if not np.isfinite(valores).all():
        raise ValueError(f"La matriz de {carpeta} contiene valores no finitos.")
    return X


def cargar_usuarios(carpeta, n_filas):
    """Verifica la correspondencia fila-ID antes de reducir o comparar matrices.

    Pre:  carpeta contiene usuarios.json y n_filas es el número de filas de su matriz.
    Post: la lista de usuarios, con fila == posición e IDs de texto no vacíos y únicos;
          ValueError si no.
    """
    archivo = Path(carpeta) / "usuarios.json"
    usuarios = json.loads(archivo.read_text(encoding="utf-8"))
    if not isinstance(usuarios, list) or len(usuarios) != n_filas:
        raise ValueError(f"{archivo} debe contener un usuario por fila de la matriz.")
    ids = []
    for i, usuario in enumerate(usuarios):
        if (not isinstance(usuario, dict) or usuario.get("fila") != i
                or not isinstance(usuario.get("usuario"), str) or not usuario["usuario"].strip()):
            raise ValueError(f"Fila o ID inválido en {archivo}, posición {i}.")
        ids.append(usuario["usuario"])
    if len(set(ids)) != n_filas:
        raise ValueError(f"Hay IDs duplicados en {archivo}.")
    return usuarios
