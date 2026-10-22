"""De un texto suelto a su cluster, con los usuarios mas parecidos dentro.

Rehace el mismo camino que los datos de entrenamiento: preproceso, TF-IDF y SVD
si la representacion lo llevaba.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse

from Clustering_jerarquico.metricas import normalizar_filas


@dataclass
class Vecino:
    usuario: str
    distancia: float
    texto: str = ""


@dataclass
class Resultado:
    cluster: int
    tamano_cluster: int
    vecinos: list[Vecino]
    tokens: list[str]
    vector: np.ndarray


def resolver_cadena(carpeta: Path) -> tuple[Path, Path | None]:
    """Devuelve (carpeta con el vectorizador, svd.joblib si lo hay).

    Una carpeta SVD no tiene vectorizador: su resumen.json apunta a la TF-IDF
    de la que salio.
    """
    carpeta = Path(carpeta)
    svd = carpeta / "svd.joblib"
    if not svd.is_file():
        if (carpeta / "X.npy").is_file():
            # Carpeta SVD sin modelo: los de 95 % de datos_completos no se suben por tamaño.
            raise FileNotFoundError(
                f"Falta {svd}. Si es un modelo grande que no está en Git, regenéralo con "
                "los comandos de resultados_tfidf/README.md; si no, ejecuta git lfs pull.")
        return carpeta, None

    resumen = json.loads((carpeta / "resumen.json").read_text(encoding="utf-8"))
    origen = resumen.get("configuracion", {}).get("entrada")
    if not origen:
        raise ValueError(f"{carpeta} tiene svd.joblib pero su resumen.json no dice de donde viene.")
    origen = Path(origen)
    # El resumen guarda la ruta de cuando se generó; si las carpetas se han movido
    # juntas (p. ej. a resultados_tfidf/datos_1_a_600), el TF-IDF es su vecina.
    if not (origen / "vectorizador.joblib").is_file():
        origen = carpeta.parent / origen.name
    return origen, svd


class Clasificador:
    """Une modelo, vectorizador y textos para responder sobre un texto nuevo."""

    def __init__(self, modelo, carpeta_representacion, csv_textos=None, preprocesador=None):
        """carpeta_representacion tiene que ser la misma con la que se entreno el modelo."""
        self.modelo = modelo
        carpeta_tfidf, ruta_svd = resolver_cadena(Path(carpeta_representacion))

        self.vectorizador = joblib.load(carpeta_tfidf / "vectorizador.joblib")
        self.svd = joblib.load(ruta_svd) if ruta_svd else None
        self.usuarios = [u["usuario"] for u in
                         json.loads((Path(carpeta_representacion) / "usuarios.json").read_text(encoding="utf-8"))]
        if len(self.usuarios) != self.modelo.X.shape[0]:
            raise ValueError(
                f"{carpeta_representacion} tiene {len(self.usuarios)} usuarios y el modelo se entreno con "
                f"{self.modelo.X.shape[0]}: no son la misma representacion (¿datos_1_a_600 frente a datos_completos?). "
                "Entrena el modelo con esa carpeta o pasa la carpeta con la que se entreno.")
        self.preprocesador = preprocesador
        self.textos = self._cargar_textos(csv_textos) if csv_textos else {}

    @staticmethod
    def _cargar_textos(ruta_csv) -> dict[str, str]:
        """Texto de cada usuario, recortado, para enseñar a los vecinos."""
        import pandas as pd

        df = pd.read_csv(ruta_csv, dtype={"subject_id": str}, keep_default_na=False)
        columna = "text" if "text" in df.columns else "text_Clustering"
        unidos = df.groupby("subject_id")[columna].apply(lambda s: " ".join(map(str, s))[:2000])
        return unidos.to_dict()

    def _motivo_descarte(self, texto: str) -> str:
        """Por que el preproceso se ha quedado sin tokens, para poder decirlo.

        Casi siempre es min_tokens: las palabras vacias se llevan medio texto y
        lo que queda no llega al minimo. El preprocesador solo devuelve None, asi
        que hay que rehacer el conteo para saberlo.
        """
        prep = self.preprocesador
        crudo = [t for t in texto.split() if t.strip()]
        if not crudo:
            return "No has escrito nada."

        sueltos = sum(1 for t in crudo
                      if t.lower().strip(".,;:!?\"'()").isalpha()
                      and t.lower() not in prep.stopwords.get("english", ()))
        if sueltos < prep.min_tokens:
            return (f"Solo quedan {sueltos} palabras con contenido y hacen falta "
                    f"{prep.min_tokens}. Las palabras vacias (I, can, the...) no cuentan: "
                    f"escribe una frase mas larga.")
        return "El texto se queda sin tokens utiles tras el preproceso."

    def vectorizar(self, texto: str):
        """Texto crudo -> vector en el mismo espacio que el entrenamiento."""
        if self.preprocesador is not None:
            tokens = self.preprocesador._procesar_texto(texto)
            if not tokens:
                raise ValueError(self._motivo_descarte(texto))
            documento = " ".join(tokens)
        else:
            tokens = texto.split()
            documento = texto

        X = self.vectorizador.transform([(documento,)])
        if self.svd is not None:
            X = self.svd.transform(X)
        if self.modelo.distancia == "cosine":
            X = normalizar_filas(X)
        return X, tokens

    def _proyeccion(self, semilla: int = 42):
        """Plano 2D de los usuarios, calculado una sola vez.

        PCA y no t-SNE: t-SNE no sabe proyectar puntos nuevos, habria que rehacer
        la nube entera en cada consulta y el mapa cambiaria de forma.
        """
        if not hasattr(self, "_pca"):
            from sklearn.decomposition import PCA

            A = self.modelo.X.toarray() if sparse.issparse(self.modelo.X) else np.asarray(self.modelo.X)
            self._pca = PCA(n_components=2, random_state=semilla)
            self._coords = self._pca.fit_transform(A)
        return self._pca, self._coords

    def grafico(self, resultado: "Resultado", ruta) -> Path:
        """Guarda un PNG situando a cada usuario por su distancia real al texto.

        Nada de proyectar a 2D: de 100 dimensiones a 2 se pierde tanto que dos
        puntos pegados en el papel pueden estar lejisimos de verdad, y entonces
        no se entiende por que los vecinos marcados son los vecinos. Aqui el eje
        X ES la distancia, asi que lo que se ve a la izquierda esta cerca y punto.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        import colorsys

        from Clustering_jerarquico.algoritmo import distancias

        pca, coords = self._proyeccion()
        vector = (resultado.vector.toarray() if sparse.issparse(resultado.vector)
                  else np.asarray(resultado.vector))
        punto = pca.transform(vector)[0]

        D = distancias(resultado.vector, self.modelo.X, self.modelo.distancia)[0]
        etiquetas = self.modelo.etiquetas
        fila = {u: i for i, u in enumerate(self.usuarios)}
        cercanos = [fila[v.usuario] for v in resultado.vecinos if v.usuario in fila]

        # Cada cluster tiene su tono, pero la saturacion la pone su distancia real
        # al texto. Asi un punto que cae cerca en el papel pero cuyo grupo esta
        # lejos de verdad sale descolorido y no engaña: la proyeccion 2D pierde
        # tanto que la posicion por si sola no vale para juzgar cercania.
        medias = {c: float(D[etiquetas == c].mean()) for c in range(self.modelo.k)}
        cerca, lejos = min(medias.values()), max(medias.values())
        margen = max(lejos - cerca, 1e-9)

        # Cada punto lleva SU propia intensidad, no la media de su grupo: dentro
        # de un mismo cluster hay gente mucho mas parecida que otra y pintarlos
        # todos igual tira esa informacion. El tono dice de que grupo es, la
        # intensidad cuanto se parece ese usuario concreto.
        suyo = D.min(), D.max()
        punto_p = (1 - (D - suyo[0]) / max(suyo[1] - suyo[0], 1e-9)) ** 0.8

        orden = sorted(medias, key=medias.get)
        orden.remove(resultado.cluster)

        def pintar(c, tono, fuerza, etiqueta, asignado=False):
            miembros = np.flatnonzero(etiquetas == c)
            # Se reescala dentro del grupo para que el degradado se note aunque
            # todos sus miembros esten lejos del texto.
            p = punto_p[miembros]
            p = (p - p.min()) / max(p.max() - p.min(), 1e-9) if len(p) > 1 else np.ones_like(p)

            if asignado:
                # Curva pronunciada: con un grupo grande casi todos los puntos
                # caen en mitad del rango y se ven iguales. Elevando a 2 solo se
                # oscurecen los que de verdad se parecen y el resto queda claro.
                q = p ** 2
                colores = [colorsys.hsv_to_rgb(tono, 0.18 + 0.72 * qi, 1.0 - 0.45 * qi) for qi in q]
                tamanos, opacidades = 12 + 48 * q, 0.40 + 0.55 * q
            else:
                colores = [colorsys.hsv_to_rgb(tono, (0.25 + 0.75 * pi) * fuerza,
                                               0.88 - 0.30 * pi * fuerza) for pi in p]
                tamanos, opacidades = 11 + 24 * p, (0.30 + 0.45 * p) * (0.55 + 0.45 * fuerza)

            ax.scatter(coords[miembros, 0], coords[miembros, 1], s=tamanos, alpha=opacidades,
                       color=colores, linewidths=0, label=etiqueta, zorder=3 if asignado else 1)

        fig, ax = plt.subplots(figsize=(10, 7))
        for i, c in enumerate(reversed(orden)):
            fuerza = 0.40 + 0.45 * (1 - (medias[c] - cerca) / margen) ** 0.8   # visibles, pero por debajo del asignado
            tono = 0.33 + 0.30 * (len(orden) - 1 - i) / max(len(orden) - 1, 1)
            # La distancia media, no un "1 - distancia" en porcentaje: tras el SVD
            # los vectores tienen componentes negativas, el coseno baja de 0 y la
            # distancia llega a 1.3, asi que ese porcentaje salia negativo con k
            # grande. La distancia se lee directa: 0 = encima, 1 = perpendicular.
            pintar(c, tono, fuerza, f"cluster {c} — dist {medias[c]:.2f}")

        pintar(resultado.cluster, 0.0, 1.0,
               f"cluster {resultado.cluster} — ASIGNADO, dist {medias[resultado.cluster]:.2f}",
               asignado=True)

        # Las lineas hacen falta porque la X casi nunca cae junto a sus vecinos:
        # el texto es una frase y los usuarios tienen cientos de publicaciones,
        # asi que PCA lo manda al borde aunque en el espacio real esten pegados.
        for i in cercanos:
            ax.plot([punto[0], coords[i, 0]], [punto[1], coords[i, 1]],
                    color="black", lw=0.9, alpha=0.45, zorder=3)
        if cercanos:
            ax.scatter(coords[cercanos, 0], coords[cercanos, 1], s=135, facecolors="none",
                       edgecolors="black", linewidths=1.8, zorder=4,
                       label=f"{len(cercanos)} mas parecidos")
        ax.scatter(*punto, s=230, marker="X", color="black", zorder=5, label="texto consultado")

        ax.set_title(f"Cluster {resultado.cluster} ({resultado.tamano_cluster} usuarios)", fontsize=13)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.5,
                  title="distancia media al texto", title_fontsize=9)
        ax.grid(alpha=0.15)
        ax.set_axisbelow(True)

        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(ruta, dpi=150)
        plt.close(fig)
        return ruta

    def consultar(self, texto: str, vecinos: int = 5) -> Resultado:
        """Asigna el texto a un cluster y busca los usuarios mas cercanos dentro."""
        X, tokens = self.vectorizar(texto)
        cluster = int(self.modelo.predecir(X)[0])

        from Clustering_jerarquico.algoritmo import distancias

        dentro = np.flatnonzero(self.modelo.etiquetas == cluster)
        D = distancias(X, self.modelo.X[dentro], self.modelo.distancia)[0]
        orden = np.argsort(D)[:vecinos]

        return Resultado(
            cluster=cluster,
            tamano_cluster=len(dentro),
            vecinos=[Vecino(usuario=self.usuarios[dentro[i]],
                            distancia=float(D[i]),
                            texto=self.textos.get(self.usuarios[dentro[i]], ""))
                     for i in orden],
            tokens=tokens,
            vector=X,
        )
