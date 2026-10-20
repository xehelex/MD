"""Informe HTML del banco de pruebas"""
from __future__ import annotations

from datetime import datetime


def _barra(valor: float, color: str) -> str:
    ancho = max(0.0, min(1.0, valor)) * 100
    return (f'<div class="barra"><div class="relleno" style="width:{ancho:.0f}%;background:{color}"></div>'
            f'<span>{valor:.3f}</span></div>')


def _color(equilibrio: float) -> str:
    if equilibrio >= 0.6:
        return "var(--bien)"
    if equilibrio >= 0.3:
        return "var(--regular)"
    return "var(--mal)"


def _veredicto(equilibrio: float, utiles: int) -> tuple[str, str]:
    if equilibrio < 0.15 or utiles <= 1:
        return "no reparte", "mal"
    if equilibrio < 0.5 or utiles < 4:
        return "flojo", "regular"
    return "reparte bien", "bien"


def construir_html(filas: list[dict], correlacion: float) -> str:
    matrices = sorted({f["matriz"] for f in filas})
    bloques = []

    for matriz in matrices:
        propias = [f for f in filas if f["matriz"] == matriz]
        combinaciones = sorted({(f["enlace"], f["distancia"]) for f in propias})

        resumen = []
        for enlace, distancia in combinaciones:
            candidatas = [f for f in propias if f["enlace"] == enlace and f["distancia"] == distancia]
            mejor = max(candidatas, key=lambda f: f["inconsistencia"]
                        if f["inconsistencia"] == f["inconsistencia"] else -1e9)
            resumen.append((enlace, distancia, mejor))

        resumen.sort(key=lambda t: t[2]["equilibrio"], reverse=True)
        ganador = resumen[0][0] if resumen else ""

        cuerpo = ""
        for enlace, distancia, mejor in resumen:
            texto, clase = _veredicto(mejor["equilibrio"], mejor["grupos_utiles"])
            corona = ' <span class="corona">mejor</span>' if enlace == ganador and clase == "bien" else ""
            cuerpo += (
                f'<tr class="{clase}">'
                f'<td><code>{enlace}</code>{corona}<br><span class="tenue">{distancia}</span></td>'
                f'<td class="num">{mejor["k"]}</td>'
                f'<td>{_barra(mejor["equilibrio"], _color(mejor["equilibrio"]))}</td>'
                f'<td class="num">{mejor["grupos_utiles"]}</td>'
                f'<td class="num">{mejor["instancias_sueltas"]}</td>'
                f'<td class="tam">{mejor["tamanos"]}</td>'
                f'<td><span class="marca {clase}">{texto}</span></td>'
                f'</tr>')

        bloques.append(f'<h2>{matriz}</h2><div class="scroll"><table>'
                       '<thead><tr><th>Enlace</th><th>k</th><th>Equilibrio</th>'
                       '<th>Grupos con masa</th><th>Sueltas</th><th>Reparto</th><th></th></tr></thead>'
                       f'<tbody>{cuerpo}</tbody></table></div>')

    aviso = ("El silhouette ordena al revés en esta rejilla, así que la decisión se toma "
             "con el equilibrio y los grupos con masa."
             if correlacion < -0.4 else
             "En esta rejilla el silhouette no contradice al equilibrio.")

    return f"""<meta charset="utf-8">
<title>Comparación de enlaces</title>
<style>
:root {{
  --fondo:#fff; --texto:#1a1d21; --suave:#6b7280; --borde:#e3e6ea; --panel:#f7f8fa;
  --bien:#16a34a; --regular:#d97706; --mal:#dc2626;
  --bien-f:#e7f5ec; --regular-f:#fdf3e3; --mal-f:#fdecec;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --fondo:#14171a; --texto:#e8eaed; --suave:#9aa3ad; --borde:#2b3036; --panel:#1b1f24;
    --bien:#4ade80; --regular:#fbbf24; --mal:#f87171;
    --bien-f:#13301f; --regular-f:#332611; --mal-f:#341818;
  }}
}}
*{{box-sizing:border-box}}
body{{margin:0;padding:2.5rem 1.25rem 4rem;background:var(--fondo);color:var(--texto);
  font:15px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:980px;margin:0 auto}}
h1{{font-size:1.6rem;margin:0 0 .3rem;letter-spacing:-.02em}}
h2{{font-size:1rem;margin:2.5rem 0 .6rem;font-family:ui-monospace,Menlo,monospace;color:var(--suave)}}
.sub{{color:var(--suave);font-size:.875rem;margin:0 0 1.5rem}}
.aviso{{padding:.9rem 1.1rem;border:1px solid var(--borde);border-radius:10px;
  background:var(--panel);font-size:.9rem;margin-bottom:.5rem}}
.scroll{{overflow-x:auto}}
table{{width:100%;border-collapse:collapse;font-size:.875rem}}
th{{text-align:left;font-weight:600;color:var(--suave);font-size:.72rem;text-transform:uppercase;
  letter-spacing:.04em;padding:0 .7rem .5rem;white-space:nowrap}}
td{{padding:.6rem .7rem;border-top:1px solid var(--borde);vertical-align:middle}}
td.num{{font-variant-numeric:tabular-nums;white-space:nowrap}}
td.tam{{font-family:ui-monospace,Menlo,monospace;font-size:.8rem;color:var(--suave);white-space:nowrap}}
code{{font:13px ui-monospace,Menlo,monospace;font-weight:600}}
.tenue{{color:var(--suave);font-size:.75rem}}
.barra{{position:relative;height:20px;min-width:150px;background:var(--panel);
  border-radius:4px;overflow:hidden}}
.relleno{{height:100%}}
.barra span{{position:absolute;right:7px;top:0;line-height:20px;font-size:.75rem;
  font-variant-numeric:tabular-nums;color:var(--texto)}}
.marca{{font-size:.75rem;font-weight:600;padding:.15rem .5rem;border-radius:5px;white-space:nowrap}}
.marca.bien{{background:var(--bien-f);color:var(--bien)}}
.marca.regular{{background:var(--regular-f);color:var(--regular)}}
.marca.mal{{background:var(--mal-f);color:var(--mal)}}
.corona{{font-size:.68rem;font-weight:700;background:var(--bien-f);color:var(--bien);
  padding:.1rem .4rem;border-radius:4px;margin-left:.4rem;text-transform:uppercase}}
</style>
<main>
<h1>Comparación de enlaces</h1>
<p class="sub">{len(filas)} configuraciones · {datetime.now():%d/%m/%Y %H:%M}</p>
<div class="aviso"><strong>Cómo leerlo.</strong> Para cada enlace se muestra el k que propone el
árbol. <strong>Equilibrio</strong> cerca de 1 significa grupos de tamaño parecido; cerca de 0, que
uno se lo come todo. <strong>Grupos con masa</strong> son los que superan el mínimo; el resto queda
como instancias sueltas. {aviso}</div>
{"".join(bloques)}
</main>
"""
