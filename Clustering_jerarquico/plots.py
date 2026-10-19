"""Gráficos del clustering jerárquico, sin librerías de clustering."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _orden_hojas(merges, n: int) -> list[int]:
    """Orden de las hojas según el recorrido del árbol, para que las ramas no se crucen."""
    hijos = {n + idx: (int(a), int(b)) for idx, (a, b, _h) in enumerate(merges)}
    pila, orden = [2 * n - 2], []
    while pila:
        nodo = pila.pop()
        if nodo < n:
            orden.append(nodo)
        else:
            izquierda, derecha = hijos[nodo]
            pila.extend((derecha, izquierda))
    return orden


def guardar_dendrograma(resultado, salida: Path):
    """Dendrograma coloreado por los grupos que quedan al nivel de corte visual.

    El corte se sitúa en el mayor salto entre alturas de fusión consecutivas.
    """
    n = resultado.n_instancias
    merges = resultado.merges

    centros = {hoja: float(pos) for pos, hoja in enumerate(_orden_hojas(merges, n))}
    alturas = {i: 0.0 for i in range(n)}
    uniones = []

    for idx, (a, b, h) in enumerate(merges):
        a, b, h = int(a), int(b), float(h)
        uniones.append((centros[a], centros[b], alturas[a], alturas[b], h))
        centros[n + idx] = (centros[a] + centros[b]) / 2
        alturas[n + idx] = h

    alturas_unicas = np.unique([u[4] for u in uniones])
    if len(alturas_unicas) >= 2:
        salto = int(np.argmax(np.diff(alturas_unicas)))
        corte = float((alturas_unicas[salto] + alturas_unicas[salto + 1]) / 2)
    elif len(alturas_unicas) == 1:
        corte = float(alturas_unicas[0]) / 2
    else:
        corte = 0.0

    padre = list(range(2 * n - 1))

    def raiz(x):
        while padre[x] != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    for idx, (a, b, h) in enumerate(merges):
        if float(h) <= corte:
            nuevo = n + idx
            padre[raiz(int(a))] = nuevo
            padre[raiz(int(b))] = nuevo

    grupos = {}
    for i in range(n):
        grupos.setdefault(raiz(i), []).append(i)

    raices_ordenadas = sorted(grupos, key=lambda r: min(grupos[r]))
    id_cluster = {r: j for j, r in enumerate(raices_ordenadas)}

    cmap = plt.get_cmap("tab20")
    colores = {c: cmap(c % 20) for c in range(len(raices_ordenadas))}

    # Una rama se pinta con el color de su grupo solo si todas sus hojas pertenecen a él.
    cluster_nodo = {i: id_cluster[raiz(i)] for i in range(n)}

    fig, ax = plt.subplots(figsize=(16, 9))

    for idx, (a, b, _h) in enumerate(merges):
        ca, cb = cluster_nodo[int(a)], cluster_nodo[int(b)]
        homogenea = ca == cb and ca != -1
        cluster_nodo[n + idx] = ca if homogenea else -1

        xa, xb, ha, hb, h = uniones[idx]
        ax.plot(
            [xa, xa, xb, xb],
            [ha, h, h, hb],
            linewidth=1.0 if homogenea else 0.8,
            color=colores[ca] if homogenea else "#777777",
            alpha=0.9,
        )

    ax.axhline(
        corte, color="red", linestyle="--", linewidth=1.5,
        label=f"Corte visual (distancia = {corte:.3g})"
    )
    ax.set_title(f"Dendrograma — {len(raices_ordenadas)} grupos al nivel de corte")
    ax.set_xlabel("Usuarios (orden del árbol)")
    ax.set_ylabel("Disimilitud de fusión")
    ax.set_xticks([])
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(salida / "dendrograma_interpretable.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def guardar_silhouette(ks, valores, salida: Path):
    """Curva del silhouette medio en función de k."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, valores, marker="o")
    ax.set_title("Silhouette medio según el número de clusters")
    ax.set_xlabel("Número de clusters (k)")
    ax.set_ylabel("Silhouette medio")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(salida / "silhouette_por_k.png", dpi=160)
    plt.close(fig)


def guardar_busqueda_k(filas, mejores: dict, titulo: str, ruta: Path):
    """Silhouette, Davies-Bouldin y Calinski-Harabasz frente a k, marcando los k visitados y el óptimo hallado.

    `filas`: diccionarios con k, silhouette, davies_bouldin, calinski_harabasz y visitado_<métrica>.
    `mejores`: {métrica: k óptimo} para las métricas sobre las que se buscó.
    """
    filas = sorted(filas, key=lambda f: f["k"])
    paneles = [
        ("silhouette", "Silhouette medio (maximizar)"),
        ("davies_bouldin", "Davies-Bouldin (minimizar)"),
        ("calinski_harabasz", "Calinski-Harabasz (maximizar)"),
    ]

    fig, ejes = plt.subplots(1, len(paneles), figsize=(19, 4.8))
    for ax, (clave, etiqueta) in zip(ejes, paneles):
        visitadas = [f for f in filas if f[f"visitado_{clave}"]]

        if len(visitadas) < len(filas):
            ax.plot([f["k"] for f in filas], [f[clave] for f in filas],
                    color="#bbbbbb", marker=".", markersize=4, linewidth=1, label="Otros k evaluados")
        if visitadas:
            ax.plot([f["k"] for f in visitadas], [f[clave] for f in visitadas],
                    color="#1f77b4", marker="o", linewidth=1, label="Visitados por la búsqueda")
        if clave in mejores:
            ax.axvline(mejores[clave], color="red", linestyle="--", label=f"k óptimo = {mejores[clave]}")

        ax.set_xlabel("Número de clusters (k)")
        ax.set_ylabel(etiqueta)
        ax.grid(alpha=0.3)
        ax.legend(fontsize="small")

    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(ruta, dpi=160)
    plt.close(fig)


def guardar_proyeccion_mds(D, etiquetas, salida: Path):
    """Proyección 2D mediante MDS clásico calculado con NumPy."""
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    D2 = D ** 2

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
