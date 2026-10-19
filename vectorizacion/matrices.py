"""Carga común de representaciones dispersas y densas y sus usuarios."""

import json
from pathlib import Path

import numpy as np
from scipy import sparse


def cargar_matriz(carpeta):
    """Lee exactamente una X.npz (CSR) o X.npy (densa), sin convertir su formato."""
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
    """Verifica la correspondencia fila-ID antes de reducir o comparar matrices."""
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
