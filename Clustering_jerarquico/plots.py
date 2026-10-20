"""Gráficas del clustering jerárquico: dendrograma, métricas por k y proyección MDS (solo NumPy y matplotlib)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")       # sin ventanas, solo se guardan los png
import matplotlib.pyplot as plt

COLORES = plt.get_cmap("tab10")


def guardar_dendrograma(resultado, etiquetas, ruta, por_cluster: int = 8, maximo: int = 24):
    """Dendrograma de una muestra de usuarios, coloreado por cluster.

    Con los ~600 usuarios no se lee nada, así que cogemos unos pocos de cada cluster y dibujamos solo las
    fusiones entre ramas que tienen alguno de ellos. Las alturas son las del árbol completo.

    Pre:  etiquetas es el corte del árbol de resultado en k clusters (2 <= k < n).
    Post: imagen guardada en ruta. La línea roja marca el corte que da los k clusters.
    """
    n, merges = resultado.n_instancias, resultado.merges

    # 1. Muestra: hasta por_cluster usuarios repartidos dentro de cada cluster (los clusters grandes primero)
    grupos = sorted(np.unique(etiquetas), key=lambda g: -np.sum(etiquetas == g))[:maximo]
    por_cluster = max(1, min(por_cluster, maximo // len(grupos)))
    muestra = set()
    for g in grupos:
        idx = np.flatnonzero(etiquetas == g)
        muestra.update(idx[np.linspace(0, len(idx) - 1, min(por_cluster, len(idx))).round().astype(int)].tolist())

    # 2. Orden de las hojas recorriendo el árbol desde la raíz, para que las ramas no se crucen
    hijos = {n + i: (int(a), int(b)) for i, (a, b, _) in enumerate(merges)}
    orden, pila = [], [2 * n - 2]
    while pila:
        nodo = pila.pop()
        if nodo >= n:
            pila.extend(reversed(hijos[nodo]))
        elif nodo in muestra:
            orden.append(nodo)

    # 3. Dibujo: para cada nodo guardamos su posición x, su altura y y su cluster (-1 si mezcla varios)
    x = {h: float(i) for i, h in enumerate(orden)}
    y = {h: 0.0 for h in orden}
    cluster = {h: int(etiquetas[h]) for h in orden}
    fig, ax = plt.subplots(figsize=(max(7.0, 0.4 * len(orden) + 2.5), 5.5))
    for i, (a, b, h) in enumerate(merges):
        a, b, nuevo = int(a), int(b), n + i
        if a in x and b in x:                   # las dos ramas tienen usuarios de la muestra: se dibuja
            c = cluster[a] if cluster[a] == cluster[b] else -1
            ax.plot([x[a], x[a], x[b], x[b]], [y[a], h, h, y[b]], color=COLORES(c % 10) if c >= 0 else "gray")
            x[nuevo], y[nuevo], cluster[nuevo] = (x[a] + x[b]) / 2, h, c
        elif a in x or b in x:                  # solo una rama: el nodo hereda la posición de esa rama
            hijo = a if a in x else b
            x[nuevo], y[nuevo], cluster[nuevo] = x[hijo], y[hijo], cluster[hijo]

    # 4. Línea de corte (entre la última fusión aplicada y la primera que no) y leyenda
    k = len(np.unique(etiquetas))
    ax.axhline((merges[n - k - 1, 2] + merges[n - k, 2]) / 2, color="red", linestyle="--", label=f"Corte en k = {k}")
    for g in sorted({cluster[h] for h in orden}):
        ax.plot([], [], color=COLORES(g % 10), linewidth=3, label=f"Cluster {g} ({np.sum(etiquetas == g)} usuarios)")
    ax.set_xticks(range(len(orden)), [str(h) for h in orden], rotation=90, fontsize=8)
    for texto, h in zip(ax.get_xticklabels(), orden):
        texto.set_color(COLORES(cluster[h] % 10))
    ax.set(title=f"Dendrograma de una muestra de {len(orden)} usuarios ({resultado.enlace}, k = {k})",
           xlabel="Índice del usuario en la matriz", ylabel="Altura de fusión")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize="small")
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)


def guardar_metricas_k(filas, mejores: dict, titulo: str, ruta):
    """Silhouette, Davies-Bouldin y Calinski-Harabasz frente a k, marcando el mejor k de cada uno.

    Pre:  filas son diccionarios con k y las tres métricas, ordenados por k; mejores es {criterio: k}.
    Post: imagen con los tres paneles guardada en ruta.
    """
    paneles = [("silhouette", "Silhouette (maximizar)"), ("davies_bouldin", "Davies-Bouldin (minimizar)"),
               ("calinski_harabasz", "Calinski-Harabasz (maximizar)")]
    fig, ejes = plt.subplots(1, 3, figsize=(17, 4.5))
    for ax, (clave, nombre) in zip(ejes, paneles):
        ax.plot([f["k"] for f in filas], [f[clave] for f in filas], marker="o")
        if clave in mejores:
            ax.axvline(mejores[clave], color="red", linestyle="--", label=f"mejor k = {mejores[clave]}")
            ax.legend()
        ax.set(xlabel="Número de clusters (k)", ylabel=nombre)
        ax.grid(alpha=0.3)
    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def guardar_mds(D, etiquetas, ruta):
    """Proyección 2D de los usuarios con MDS clásico, coloreada por cluster.

    MDS clásico: B = -1/2 · J·D²·J (J centra la matriz) y las coordenadas son los dos autovectores
    con mayor autovalor, escalados por su raíz. Se hace a mano con np.linalg.eigh.

    Pre:  D es la matriz (n, n) de distancias y etiquetas tiene n elementos.
    Post: imagen guardada en ruta.
    """
    n = len(D)
    J = np.eye(n) - 1.0 / n
    valores, vectores = np.linalg.eigh(-0.5 * J @ (D ** 2) @ J)   # eigh los da en orden creciente
    coords = vectores[:, [-1, -2]] * np.sqrt(np.maximum(valores[[-1, -2]], 0))
    fig, ax = plt.subplots(figsize=(9, 7))
    for g in np.unique(etiquetas):
        ax.scatter(*coords[etiquetas == g].T, s=28, alpha=0.75, label=f"Cluster {g}")
    ax.set(title="Proyección 2D (MDS) a partir de las distancias", xlabel="MDS 1", ylabel="MDS 2")
    ax.legend(fontsize="small")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
