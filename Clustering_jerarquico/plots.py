"""Generación de gráficos reproducibles sin recurrir a librerías de clustering."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def guardar_dendrograma(resultado, salida: Path, max_etiquetas: int = 60):
    """Dibuja un dendrograma a mano; ajusta el tamaño según la cantidad de instancias."""
    n = resultado.n_instancias

    # Coordenadas horizontales para ir centrando los nodos a medida que se unen
    centros = {i: float(i) for i in range(n)}
    alturas = {i: 0.0 for i in range(n)}

    fig, ax = plt.subplots(figsize=(max(12, min(24, n * 0.035)), 7))
    for idx, (a, b, h) in enumerate(resultado.merges):
        a, b, nuevo = int(a), int(b), n + idx
        xa, xb = centros[a], centros[b]
        ha, hb = alturas[a], alturas[b]

        ax.plot([xa, xa, xb, xb], [ha, h, h, hb], linewidth=0.65)
        centros[nuevo] = (xa + xb) / 2
        alturas[nuevo] = h

    ax.set_title(f"Dendrograma — enlace={resultado.enlace}, distancia={resultado.distancia}")
    ax.set_xlabel("Instancias (orden interno del algoritmo)")
    ax.set_ylabel("Disimilitud de fusión")
    ax.set_xticks([])
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(salida / "dendrograma.png", dpi=160)
    plt.close(fig)


def guardar_silhouette(ks, valores, salida: Path):
    """Guarda la curva del silhouette medio en función del número de clusters k."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, valores, marker="o")
    ax.set_title("Silhouette medio según el número de clusters")
    ax.set_xlabel("Número de clusters (k)")
    ax.set_ylabel("Silhouette medio")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(salida / "silhouette_por_k.png", dpi=160)
    plt.close(fig)


def guardar_proyeccion_mds(D, etiquetas, salida: Path):
    """Genera una proyección 2D usando MDS clásico calculado directamente con NumPy."""
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    D2 = D ** 2

    # Matriz de centrado para proyectar a 2D
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J

    valores, vectores = np.linalg.eigh(B)
    orden = np.argsort(valores)[::-1][:2]
    vals = np.maximum(valores[orden], 0.0)
    coords = vectores[:, orden] * np.sqrt(vals)

    labels = np.asarray(etiquetas)
    fig, ax = plt.subplots(figsize=(9, 7))
    for g in np.unique(labels):
        mask = labels == g
        ax.scatter(coords[mask, 0], coords[mask, 1], s=28, alpha=0.75, label=f"Cluster {g}")

    ax.set_title("Proyección 2D MDS a partir de las distancias")
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.legend(loc="best", fontsize="small")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(salida / "clusters_mds_2d.png", dpi=160)
    plt.close(fig)