"""Gráficos del clustering jerárquico (solo NumPy y matplotlib)."""
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


def elegir_muestra(D: np.ndarray, etiquetas: np.ndarray, por_cluster: int = 8, maximo: int = 24) -> np.ndarray:
    """Índices de las instancias que se dibujan en el dendrograma.

    De cada cluster se toman `por_cluster` instancias repartidas por igual entre la más central
    (menor distancia media al resto de su cluster) y la más periférica. Si hay muchos clusters, se
    reduce el número por cluster para no superar `maximo`; si aun así no caben, se omiten los más pequeños.

    Pre:  D es la matriz (n, n) de distancias y etiquetas tiene n elementos.
    Post: vector ordenado de índices sin repetir, con al menos una instancia por cluster dibujado.
    """
    etiquetas = np.asarray(etiquetas)
    grupos = sorted(np.unique(etiquetas), key=lambda g: -np.sum(etiquetas == g))[:maximo]
    por_cluster = max(1, min(por_cluster, maximo // len(grupos)))

    muestra = []
    for g in grupos:
        idx = np.flatnonzero(etiquetas == g)
        if len(idx) <= por_cluster:
            muestra.extend(idx)
            continue
        centralidad = D[np.ix_(idx, idx)].mean(axis=1)
        de_centro_a_borde = idx[np.argsort(centralidad, kind="stable")]
        posiciones = np.round(np.linspace(0, len(idx) - 1, por_cluster)).astype(int)
        muestra.extend(de_centro_a_borde[posiciones])
    return np.sort(np.asarray(muestra, dtype=int))


def recortar_arbol(merges, n: int, muestra) -> np.ndarray:
    """Árbol de fusiones restringido a las hojas de `muestra`, con las alturas originales.

    Pre:  merges es el árbol completo de n hojas y muestra contiene índices de hojas distintos.
    Post: matriz (m-1, 3) con el mismo formato que `merges` pero con m = len(muestra) hojas
          (ids 0..m-1, en el orden de `muestra`). Una fusión se conserva solo si junta ramas con hojas de la muestra.
    """
    m = len(muestra)
    nodo = {int(hoja): j for j, hoja in enumerate(muestra)}
    recortado, siguiente = [], m
    for idx, (a, b, altura) in enumerate(merges):
        na, nb = nodo.get(int(a)), nodo.get(int(b))
        if na is not None and nb is not None:
            recortado.append((na, nb, float(altura)))
            nodo[n + idx] = siguiente
            siguiente += 1
        elif na is not None or nb is not None:
            nodo[n + idx] = na if na is not None else nb
    return np.asarray(recortado, dtype=float)


def guardar_dendrograma(resultado, etiquetas, D, salida: Path, por_cluster: int = 8, maximo: int = 24):
    """Dendrograma de una muestra pequeña de usuarios, coloreado por cluster.

    Dibujar los n usuarios produce un gráfico ilegible, así que se eligen unos pocos de cada cluster
    (ver `elegir_muestra`) y se recorta el árbol completo a ellos: las alturas son las del clustering real.
    La línea roja marca el corte que da los k clusters de `etiquetas`.

    Pre:  etiquetas es un corte del árbol de `resultado` en k clusters (2 <= k < n) y D es la matriz (n, n)
          con la que se mide la centralidad de cada usuario.
    Post: escribe salida/dendrograma_muestra.png.
    """
    etiquetas = np.asarray(etiquetas)
    n = resultado.n_instancias
    grupos = list(np.unique(etiquetas))
    k = len(grupos)

    muestra = elegir_muestra(D, etiquetas, por_cluster, maximo)
    m = len(muestra)
    if m < 2:
        return
    arbol = recortar_arbol(resultado.merges, n, muestra)

    alturas_totales = resultado.merges[:, 2]
    corte = float((alturas_totales[n - k - 1] + alturas_totales[n - k]) / 2)

    cmap = plt.get_cmap("tab10")
    colores = {g: cmap(i % 10) for i, g in enumerate(grupos)}
    orden = _orden_hojas(arbol, m)
    centros = {hoja: float(pos) for pos, hoja in enumerate(orden)}
    alturas = {i: 0.0 for i in range(m)}
    cluster_de = {i: int(etiquetas[muestra[i]]) for i in range(m)}

    fig, ax = plt.subplots(figsize=(max(7.0, 0.4 * m + 2.5), 5.5))
    for idx, (a, b, altura) in enumerate(arbol):
        a, b = int(a), int(b)
        mismo = cluster_de[a] == cluster_de[b] and cluster_de[a] != -1
        ax.plot([centros[a], centros[a], centros[b], centros[b]], [alturas[a], altura, altura, alturas[b]],
                color=colores[cluster_de[a]] if mismo else "#777777", linewidth=1.5)
        nuevo = m + idx
        cluster_de[nuevo] = cluster_de[a] if mismo else -1
        centros[nuevo] = (centros[a] + centros[b]) / 2
        alturas[nuevo] = float(altura)

    ax.axhline(corte, color="red", linestyle="--", linewidth=1.2, label=f"Corte en k = {k}")
    for g in sorted(set(etiquetas[muestra])):
        tam = int(np.sum(etiquetas == g))
        ax.plot([], [], color=colores[g], linewidth=3, label=f"Cluster {g} ({tam} {'usuario' if tam == 1 else 'usuarios'})")

    ax.set_xticks(range(m))
    ax.set_xticklabels([str(muestra[hoja]) for hoja in orden], rotation=90, fontsize=8)
    for texto, hoja in zip(ax.get_xticklabels(), orden):
        texto.set_color(colores[cluster_de[hoja]])
    ax.set_title(f"Dendrograma de una muestra de {m} usuarios ({resultado.enlace}, k = {k})")
    ax.set_xlabel("Índice del usuario en la matriz\n"
                  "Muestra repartida de más central a más periférico en cada cluster; alturas del árbol completo",
                  fontsize=8)
    ax.set_ylabel("Altura de fusión")
    ax.grid(axis="y", alpha=0.2)
    ax.margins(y=0.06)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize="small")
    fig.tight_layout()
    fig.savefig(salida / "dendrograma_muestra.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def guardar_silhouette(ks, valores, salida: Path):
    """Curva del silhouette medio en función de k (salida/silhouette_por_k.png)."""
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
    """Silhouette, Davies-Bouldin y Calinski-Harabasz frente a k, marcando los k visitados y el óptimo.

    Pre:  `filas` son diccionarios con k, las tres métricas y visitado_<métrica>;
          `mejores` es {métrica: k óptimo} para las métricas sobre las que se buscó.
    Post: escribe la figura en `ruta`.
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
    """Proyección 2D de las instancias con MDS clásico (descomposición propia con NumPy).

    Pre:  D es la matriz (n, n) de distancias y etiquetas tiene n elementos.
    Post: escribe salida/clusters_mds_2d.png.
    """
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
