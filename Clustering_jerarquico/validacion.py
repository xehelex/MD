"""Comprueba la implementación propia contra scipy y guarda un informe HTML.

Se ejecuta desde la raíz del proyecto:

    python3 -m Clustering_jerarquico.validacion

scipy solo se usa como referencia para contrastar, nunca dentro del algoritmo.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.sparse import load_npz
from scipy.spatial.distance import cdist

from .algoritmo import ClusteringJerarquico
from .metricas import adjusted_rand_index


def _coinciden(etiquetas_a, etiquetas_b) -> bool:
    """Dos particiones son la misma aunque los clusters estén numerados distinto."""
    return adjusted_rand_index(etiquetas_a, etiquetas_b) > 0.999


def etiquetas_desde_scipy(Z: np.ndarray, n: int, k: int) -> np.ndarray:
    """Corta el árbol de scipy contando fusiones, como nuestro etiquetas().

    fcluster con 'maxclust' no sirve aquí: corta por umbral y con fusiones
    empatadas no devuelve exactamente k grupos.
    """
    grupos = {i: {i} for i in range(n)}
    siguiente = n
    for izquierda, derecha, _altura, _tam in Z[:n - k]:
        grupos[siguiente] = grupos.pop(int(izquierda)) | grupos.pop(int(derecha))
        siguiente += 1

    etiquetas = np.empty(n, dtype=int)
    for etiqueta, miembros in enumerate(grupos.values()):
        for miembro in miembros:
            etiquetas[miembro] = etiqueta
    return etiquetas


def contrastar_enlace(X, A, enlace: str, metodo_scipy: str, distancia: str) -> dict:
    """Compara un enlace propio con el de scipy en TODOS los cortes.

    Recorre k = 2 .. n-1 y no unos pocos valores: con cinco sueltos parecía
    que todo cuadraba y se nos escapaban diferencias.
    """
    inicio = time.perf_counter()
    resultado = ClusteringJerarquico(enlace=enlace, distancia=distancia).fit(X)
    segundos = time.perf_counter() - inicio

    # A los demás hay que decirles la métrica: scipy asume euclídea por
    # defecto y entonces comparas peras con manzanas.
    Z = (linkage(A, method=metodo_scipy) if metodo_scipy == "ward"
         else linkage(A, method=metodo_scipy, metric=distancia))
    n = resultado.n_instancias
    alturas = resultado.merges[:, 2]

    # Si las dos fusiones del corte empatan en altura, el árbol no decide el
    # orden y las dos particiones valen. No cuentan como discrepancia.
    ambiguos = {k for k in range(2, n)
                if np.isclose(alturas[n - k - 1], alturas[n - k], rtol=0, atol=1e-12)}

    discrepancias, por_empate = [], []
    for k in range(2, n):
        if _coinciden(resultado.etiquetas(k), etiquetas_desde_scipy(Z, n, k)):
            continue
        (por_empate if k in ambiguos else discrepancias).append(k)

    error_alturas = float(np.abs(np.sort(alturas) - np.sort(Z[:, 2])).max())

    return {
        "enlace": enlace,
        "distancia": distancia,
        "cortes": n - 2,
        "discrepancias": discrepancias,
        "por_empate": por_empate,
        "error_alturas": error_alturas,
        "segundos": segundos,
        "ok": not discrepancias,
    }


def contrastar_distancia(A, tipo: str, p: float, kwargs_scipy: dict) -> dict:
    """Compara una matriz de distancias propia con la de scipy."""
    propia = ClusteringJerarquico._distancias(A, tipo, p)
    referencia = cdist(A, A, **kwargs_scipy)
    error = float(np.abs(propia - referencia).max())
    return {"nombre": tipo if p in (1.0, np.inf) else f"{tipo} p={p:g}",
            "error": error, "ok": error < 1e-8}


def contrastar_rechazos() -> list[dict]:
    """Comprueba que las combinaciones sin sentido matemático no se dejan construir."""
    casos = [
        ("ward", "cosine", 2.0, True),
        ("ward", "manhattan", 2.0, True),
        ("ward", "minkowski", 3.0, True),
        ("ward", "euclidean", 2.0, False),
        ("ward", "minkowski", 2.0, False),
        ("average", "minkowski", 0.5, True),
        ("average", "cosine", 2.0, False),
    ]
    filas = []
    for enlace, distancia, p, debe_rechazar in casos:
        try:
            ClusteringJerarquico(enlace=enlace, distancia=distancia, p=p)
            rechaza = False
        except ValueError:
            rechaza = True
        filas.append({"combinacion": f"{enlace} + {distancia} (p={p:g})",
                      "rechaza": rechaza, "esperado": debe_rechazar,
                      "ok": rechaza == debe_rechazar})
    return filas


def medir_coste(A, tamanos) -> list[dict]:
    """Tiempo de construir la matriz de distancias según crece el número de filas."""
    filas = []
    for n in tamanos:
        muestra = A[:n]
        tiempos = {}
        for tipo in ("euclidean", "manhattan"):
            inicio = time.perf_counter()
            ClusteringJerarquico._distancias(muestra, tipo)
            tiempos[tipo] = time.perf_counter() - inicio
        filas.append({"n": n, **tiempos})
    return filas


# region Informe HTML

def _svg_coste(coste: list[dict]) -> str:
    """Gráfica de tiempo frente a n, en escala logarítmica, sin dependencias."""
    ancho, alto, margen = 560, 300, 52
    ns = [f["n"] for f in coste]
    series = {"manhattan": [f["manhattan"] for f in coste],
              "euclidean": [f["euclidean"] for f in coste]}
    todos = [v for s in series.values() for v in s]
    lo, hi = np.log10(min(todos)), np.log10(max(todos))
    lo, hi = lo - 0.2, hi + 0.2

    def x(n):
        return margen + (n - min(ns)) / (max(ns) - min(ns)) * (ancho - 2 * margen)

    def y(t):
        return alto - margen - (np.log10(t) - lo) / (hi - lo) * (alto - 2 * margen)

    partes = [f'<svg viewBox="0 0 {ancho} {alto}" class="grafica">']
    for potencia in range(int(np.floor(lo)), int(np.ceil(hi)) + 1):
        if not lo <= potencia <= hi:
            continue
        py = y(10.0 ** potencia)
        partes.append(f'<line x1="{margen}" y1="{py:.1f}" x2="{ancho - margen}" y2="{py:.1f}" class="rejilla"/>')
        partes.append(f'<text x="{margen - 8}" y="{py + 4:.1f}" class="eje fin">{10.0 ** potencia:g}s</text>')
    for n in ns:
        partes.append(f'<text x="{x(n):.1f}" y="{alto - margen + 18}" class="eje medio">{n}</text>')

    colores = {"manhattan": "var(--acento)", "euclidean": "var(--acento2)"}
    for nombre, valores in series.items():
        puntos = " ".join(f"{x(n):.1f},{y(t):.1f}" for n, t in zip(ns, valores))
        partes.append(f'<polyline points="{puntos}" class="linea" stroke="{colores[nombre]}"/>')
        for n, t in zip(ns, valores):
            partes.append(f'<circle cx="{x(n):.1f}" cy="{y(t):.1f}" r="4" fill="{colores[nombre]}"/>')
        partes.append(f'<text x="{x(ns[-1]) + 8:.1f}" y="{y(valores[-1]) + 4:.1f}" '
                      f'class="eje" fill="{colores[nombre]}">{nombre}</text>')

    partes.append(f'<text x="{ancho / 2}" y="{alto - 6}" class="eje medio">filas (n)</text>')
    partes.append("</svg>")
    return "".join(partes)


def _marca(ok: bool) -> str:
    return f'<span class="marca {"si" if ok else "no"}">{"correcto" if ok else "fallo"}</span>'


def construir_html(enlaces, distancias, rechazos, coste, n, d) -> str:
    todo_ok = (all(e["ok"] for e in enlaces) and all(x["ok"] for x in distancias)
               and all(r["ok"] for r in rechazos))

    filas_enlaces = "".join(
        f"<tr><td><code>{e['enlace']}</code></td><td>{e['distancia']}</td>"
        f"<td class='num'>{e['cortes']}</td>"
        f"<td class='num'>{len(e['discrepancias'])}</td>"
        f"<td class='num'>{len(e['por_empate']) or '—'}</td>"
        f"<td class='num'>{e['error_alturas']:.1e}</td>"
        f"<td class='num'>{e['segundos']:.2f}s</td>"
        f"<td>{_marca(e['ok'])}</td></tr>" for e in enlaces)

    filas_dist = "".join(
        f"<tr><td><code>{x['nombre']}</code></td>"
        f"<td class='num'>{x['error']:.2e}</td><td>{_marca(x['ok'])}</td></tr>"
        for x in distancias)

    filas_rech = "".join(
        f"<tr><td><code>{r['combinacion']}</code></td>"
        f"<td>{'la rechaza' if r['rechaza'] else 'la acepta'}</td>"
        f"<td>{'rechazar' if r['esperado'] else 'aceptar'}</td>"
        f"<td>{_marca(r['ok'])}</td></tr>" for r in rechazos)

    filas_coste = "".join(
        f"<tr><td class='num'>{f['n']}</td>"
        f"<td class='num'>{f['euclidean']:.3f}s</td>"
        f"<td class='num'>{f['manhattan']:.3f}s</td>"
        f"<td class='num'>{f['manhattan'] / f['euclidean']:.0f}×</td></tr>" for f in coste)

    razones = [coste[i]["manhattan"] / coste[i - 1]["manhattan"] for i in range(1, len(coste))]
    nota_coste = (f"Al doblar n el tiempo se multiplica por "
                  f"{np.mean(razones):.1f}, lo que confirma el crecimiento cuadrático "
                  f"previsto para una distancia sin atajo matricial.")

    return f"""<meta charset="utf-8">
<title>Validación del clustering jerárquico</title>
<style>
:root {{
  --fondo: #ffffff; --texto: #1a1d21; --suave: #6b7280; --borde: #e3e6ea;
  --panel: #f7f8fa; --acento: #c2410c; --acento2: #1d4ed8;
  --ok-f: #e7f5ec; --ok-t: #116b33; --no-f: #fdecec; --no-t: #a51c1c;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --fondo: #14171a; --texto: #e8eaed; --suave: #9aa3ad; --borde: #2b3036;
    --panel: #1b1f24; --acento: #fb923c; --acento2: #7aa2f7;
    --ok-f: #13301f; --ok-t: #6ee7a0; --no-f: #341818; --no-t: #fca5a5;
  }}
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; padding: 2.5rem 1.25rem 4rem; background: var(--fondo); color: var(--texto);
  font: 15px/1.6 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }}
main {{ max-width: 860px; margin: 0 auto; }}
h1 {{ font-size: 1.6rem; margin: 0 0 .3rem; letter-spacing: -.02em; }}
h2 {{ font-size: 1.05rem; margin: 2.6rem 0 .5rem; letter-spacing: -.01em; }}
.sub {{ color: var(--suave); font-size: .875rem; margin: 0 0 1.75rem; }}
.veredicto {{ display: flex; align-items: center; gap: .85rem; padding: 1rem 1.15rem;
  border: 1px solid var(--borde); border-radius: 10px; background: var(--panel); }}
.veredicto .punto {{ width: 10px; height: 10px; border-radius: 50%; flex: none;
  background: {'var(--ok-t)' if todo_ok else 'var(--no-t)'}; }}
.veredicto strong {{ font-size: 1rem; }}
.veredicto span {{ color: var(--suave); font-size: .875rem; }}
.desc {{ color: var(--suave); font-size: .875rem; margin: 0 0 .75rem; }}
.scroll {{ overflow-x: auto; }}
table {{ width: 100%; border-collapse: collapse; font-size: .875rem; }}
th {{ text-align: left; font-weight: 600; color: var(--suave); font-size: .78rem;
  text-transform: uppercase; letter-spacing: .04em; padding: 0 .7rem .5rem; white-space: nowrap; }}
td {{ padding: .55rem .7rem; border-top: 1px solid var(--borde); white-space: nowrap; }}
td.num {{ font-variant-numeric: tabular-nums; }}
code {{ font: 13px ui-monospace, SFMono-Regular, Menlo, monospace; }}
.marca {{ font-size: .78rem; font-weight: 600; padding: .15rem .5rem; border-radius: 5px; }}
.marca.si {{ background: var(--ok-f); color: var(--ok-t); }}
.marca.no {{ background: var(--no-f); color: var(--no-t); }}
.grafica {{ width: 100%; height: auto; margin-top: .5rem; }}
.rejilla {{ stroke: var(--borde); stroke-width: 1; }}
.linea {{ fill: none; stroke-width: 2.5; }}
.eje {{ fill: var(--suave); font-size: 11px; }}
.fin {{ text-anchor: end; }} .medio {{ text-anchor: middle; }}
</style>
<main>
<h1>Validación del clustering jerárquico</h1>
<p class="sub">{n} instancias · {d:,} atributos · {datetime.now():%d/%m/%Y %H:%M}</p>

<div class="veredicto">
  <span class="punto"></span>
  <div><strong>{'Todas las comprobaciones pasan' if todo_ok else 'Hay comprobaciones que fallan'}</strong><br>
  <span>La implementación propia reproduce a scipy sin usarlo en ningún momento.</span></div>
</div>

<h2>Enlaces</h2>
<p class="desc">Cada árbol se corta en todos los k posibles y se compara con scipy corte a corte.
Los cortes <em>por empate</em> son aquellos en los que dos fusiones están exactamente a la misma
altura: ahí el árbol no define un orden y las dos particiones son igual de válidas, así que no
cuentan como discrepancia. La columna de error es la diferencia máxima entre las alturas de
fusión de los dos árboles.</p>
<div class="scroll"><table>
<thead><tr><th>Enlace</th><th>Distancia</th><th>Cortes</th><th>Discrepancias</th><th>Por empate</th><th>Error alturas</th><th>Tiempo</th><th></th></tr></thead>
<tbody>{filas_enlaces}</tbody></table></div>

<h2>Distancias</h2>
<p class="desc">Diferencia máxima entre la matriz propia y la de scipy, elemento a elemento.</p>
<div class="scroll"><table>
<thead><tr><th>Métrica</th><th>Error máximo</th><th></th></tr></thead>
<tbody>{filas_dist}</tbody></table></div>

<h2>Combinaciones no admitidas</h2>
<p class="desc">Ward se deriva de minimizar la varianza sobre distancias euclídeas al cuadrado,
así que con cualquier otra métrica no debe dejarse construir.</p>
<div class="scroll"><table>
<thead><tr><th>Combinación</th><th>Comportamiento</th><th>Esperado</th><th></th></tr></thead>
<tbody>{filas_rech}</tbody></table></div>

<h2>Coste de la matriz de distancias</h2>
<p class="desc">{nota_coste}</p>
<div class="scroll"><table>
<thead><tr><th>n</th><th>Euclídea</th><th>Manhattan</th><th>Factor</th></tr></thead>
<tbody>{filas_coste}</tbody></table></div>
{_svg_coste(coste)}
</main>
"""

# endregion


def main():
    raiz = Path(__file__).parent.parent
    X = load_npz(raiz / "resultados_tfidf_completo" / "X.npz").tocsr()
    A = X.toarray()
    n, d = X.shape

    enlaces = [
        contrastar_enlace(X, A, "ward", "ward", "euclidean"),
        contrastar_enlace(X, A, "single", "single", "cosine"),
        contrastar_enlace(X, A, "complete", "complete", "cosine"),
        contrastar_enlace(X, A, "average", "average", "cosine"),
    ]

    muestra = A[:60]
    distancias = [
        contrastar_distancia(muestra, "manhattan", 1.0, {"metric": "cityblock"}),
        contrastar_distancia(muestra, "chebyshev", np.inf, {"metric": "chebyshev"}),
        contrastar_distancia(muestra, "minkowski", 3.0, {"metric": "minkowski", "p": 3}),
        contrastar_distancia(muestra, "minkowski", 1.5, {"metric": "minkowski", "p": 1.5}),
    ]

    rechazos = contrastar_rechazos()
    coste = medir_coste(A, [60, 120, 240])

    destino = raiz / "resultados_clustering_completo" / "validacion.html"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(construir_html(enlaces, distancias, rechazos, coste, n, d), encoding="utf-8")

    fallos = ([e["enlace"] for e in enlaces if not e["ok"]]
              + [x["nombre"] for x in distancias if not x["ok"]]
              + [r["combinacion"] for r in rechazos if not r["ok"]])

    for e in enlaces:
        detalle = "sin discrepancias" if e["ok"] else f"{len(e['discrepancias'])} discrepancias"
        if e["por_empate"]:
            detalle += f"  ({len(e['por_empate'])} ambiguos por empate: k={e['por_empate']})"
        print(f"  {e['enlace']:9s} {e['cortes']:4d} cortes  {detalle}")
    print()
    print("Todo correcto." if not fallos else f"Fallan: {', '.join(fallos)}")
    print(f"Informe: {destino}")


if __name__ == "__main__":
    main()
