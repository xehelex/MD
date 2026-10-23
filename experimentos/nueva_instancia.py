"""De un texto suelto a su cluster, con los usuarios mas parecidos dentro.

Rehace el mismo camino que los datos de entrenamiento: preproceso, TF-IDF y SVD
si la representacion lo llevaba.
"""
from __future__ import annotations

import collections
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
    votos: tuple[int, int] | None = None     # (a favor, total), solo si fue por vecinos
    ruta: str = ""                           # "umap" o "vecinos"; vacio en TF-IDF

    def aviso(self) -> str:
        """Por que no fiarse de esta asignacion. Solo en la densa.

        Un vector de usuario es la media de cientos de posts, asi que una frase
        corta se parece mas al ruido general que a un tema. En TF-IDF el minimo
        ya lo pone el preprocesador.

        Pre:  ninguna; en TF-IDF (ruta vacia) siempre devuelve "".
        Post: el motivo de desconfiar, o "" si no hay ninguno.
        """
        if not self.ruta:
            return ""
        if len(self.tokens) < 10:
            return (f"Solo {len(self.tokens)} palabras. Por debajo de 10 la asignacion "
                    "no es de fiar: escribe una frase mas larga.")
        if self.votos and self.votos[0] * 2 <= self.votos[1]:
            return (f"Votacion repartida ({self.votos[0]} de {self.votos[1]}): "
                    "el texto esta entre dos grupos.")
        return ""


def resolver_cadena(carpeta: Path) -> tuple[Path, Path | None]:
    """Devuelve (carpeta con el vectorizador, svd.joblib si lo hay).

    Una carpeta SVD no tiene vectorizador: el resumen.json dice de donde viene. Si no hay svd.joblib, devuelve None.

    Pre:  la carpeta existe y, si lleva svd.joblib, su resumen.json dice de
          donde viene.
    Post: (carpeta con vectorizador.joblib, ruta del svd o None).
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
    # hay que buscarla en la misma carpeta que la SVD.
    if not (origen / "vectorizador.joblib").is_file():
        origen = carpeta.parent / origen.name
    return origen, svd


def es_densa(carpeta: Path) -> bool:
    """Carpeta de la cadena Roberta: su config.json dice con que modelo se embebio.

    Pre:  carpeta es una ruta, exista o no.
    Post: True solo si hay config.json con model_name. Una carpeta TF-IDF da
          False porque su resumen.json no lleva ese campo.
    """
    config = Path(carpeta) / "config.json"
    return config.is_file() and "model_name" in json.loads(config.read_text(encoding="utf-8"))


def embeddings_crudos(carpeta: Path) -> tuple[np.ndarray, list[str]]:
    """Los vectores de 1024 dimensiones de los que salio la reduccion.

    El user_embeddings.npz de una carpeta PCA_umap_* ya viene reducido.

    Pre:  existe la hermana sin_reduccion con su user_embeddings.npz.
    Post: (matriz (n, 1024), lista de n usuarios en el orden de las filas).
    """
    carpeta = Path(carpeta)
    base = carpeta if carpeta.name == "sin_reduccion" else carpeta.parent / "sin_reduccion"
    fichero = base / "user_embeddings.npz"
    if not fichero.is_file():
        raise FileNotFoundError(
            f"No existe {fichero}. Hace falta para consultar textos nuevos sobre la "
            f"representacion densa: generalo con python -m vectorizacion_densa.")
    datos = np.load(fichero)
    return datos["embeddings"], [str(u) for u in datos["subject_id"]]


def recompilar_distancias(reductor):
    """Cambia las distancias de numba que UMAP guarda en el .joblib por las de este Python.

    El pickle lleva el bytecode con el que se compilaron (3.12) y con otras
    versiones numba falla al transformar. Son funciones de umap.distances, asi
    que basta con volver a cogerlas de ahi: lo ajustado (grafo, embedding,
    escalado y PCA) no cambia.

    Pre:  reductor es el Pipeline escalado -> PCA -> UMAP de reduccionDim.
    Post: el mismo reductor, con _input/_inverse/_output_distance_func de la
          version instalada de umap. Lo que no se encuentre se deja como venia.
    """
    import umap.distances as distancias_umap

    umap_ = reductor.steps[-1][1] if hasattr(reductor, "steps") else reductor
    for atributo in ("_input_distance_func", "_inverse_distance_func", "_output_distance_func"):
        nombre = getattr(getattr(umap_, atributo, None), "__qualname__", "")
        nueva = getattr(distancias_umap, nombre, None) if nombre else None
        if nueva is not None:
            setattr(umap_, atributo, nueva)
    return reductor


def cargar_reductor(carpeta: Path):
    """El Pipeline escalado -> PCA -> UMAP, si este Python puede abrirlo.

    Lleva bytecode de numba dentro, generado con 3.12. Al abrirlo se recompilan
    sus distancias (recompilar_distancias), asi que tambien transforma con 3.13;
    3.10 ni siquiera lo puede cargar.

    Pre:  carpeta es una ruta, exista o no el reductor.
    Post: (reductor, "") si se pudo abrir, (None, motivo) si no. Nunca lanza:
          que falte o no abra es un caso previsto, no un error.
    """
    ruta = Path(carpeta) / "reductor.joblib"
    if not ruta.is_file():
        return None, f"no existe {ruta.name}"
    # Abrirlo importa umap, que si encuentra tensorflow instalado se trae su
    # parametric_umap entero y suelta un aviso de CPU que no viene a cuento.
    import os

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    try:
        return recompilar_distancias(joblib.load(ruta)), ""
    except Exception as error:       # numba falla distinto segun la version
        return None, f"{type(error).__name__}: {str(error)[:70]}"


class EmbebedorRoberta:
    """Texto suelto -> vector de Roberta, igual que se hizo al entrenar.

    Tira de vectorizacion_densa en vez de copiar la limpieza, para que no se
    queden dos versiones distintas.
    """

    def __init__(self, model_name: str, max_length: int = 512):
        """Carga el tokenizador y el modelo.

        Pre:  model_name es un modelo de Hugging Face y max_length >= 1.
        Post: embebedor listo. La primera vez descarga el modelo, que en
              roberta-large son 1,4 GB.
        """
        import transformers

        from vectorizacion_densa.roberta import RobertaPorUsuario

        # Calla el informe de pesos: roberta-large trae cabeza de lenguaje y
        # aqui solo se usa el cuerpo.
        transformers.logging.set_verbosity_error()
        # data_path no se usa: solo queremos el tokenizador y el modelo.
        self.roberta = RobertaPorUsuario(model_name=model_name, max_length=max_length)

    def __call__(self, texto: str) -> np.ndarray:
        """Pre: texto no vacio.  Post: vector (1, d). Lanza ValueError si la
        limpieza lo deja sin nada (un texto que solo es un enlace)."""
        from tqdm import tqdm

        limpio = self.roberta.clean_text(texto)
        if not limpio:
            raise ValueError("El texto se queda vacio al limpiarlo (URLs y menciones se quitan).")
        with tqdm(total=1, disable=True) as barra:
            return self.roberta._encode_chunk([limpio], barra)


class Clasificador:
    """Une modelo, vectorizador y textos para responder sobre un texto nuevo."""

    def __init__(self, modelo, carpeta_representacion, csv_textos=None, preprocesador=None):
        """carpeta_representacion tiene que ser la misma con la que se entreno el modelo.

        Pre:  la carpeta tiene usuarios.json y, segun el caso, vectorizador
              o config.json de Roberta. En TF-IDF hace falta preprocesador.
        Post: clasificador listo para consultar. Lanza ValueError si el modelo
              no cuadra con la representacion, en numero de usuarios o de
              dimensiones.
        """
        self.modelo = modelo
        carpeta = Path(carpeta_representacion)
        self.densa = es_densa(carpeta)

        if self.densa:
            # Repetir la cadena entera acierta el 95% frente al 87% de ir por
            # vecinos; los vecinos son el respaldo si el reductor no abre.
            config = json.loads((carpeta / "config.json").read_text(encoding="utf-8"))
            self.vectorizador = self.svd = None
            self.embeddings, usuarios_crudos = embeddings_crudos(carpeta)
            self.reductor, self.sin_reductor = cargar_reductor(carpeta)
            self.embebedor = EmbebedorRoberta(config["model_name"],
                                              config.get("max_length", 512))
        else:
            carpeta_tfidf, ruta_svd = resolver_cadena(carpeta)
            self.vectorizador = joblib.load(carpeta_tfidf / "vectorizador.joblib")
            self.svd = joblib.load(ruta_svd) if ruta_svd else None

        self.usuarios = [u["usuario"] for u in
                         json.loads((carpeta / "usuarios.json").read_text(encoding="utf-8"))]
        if self.densa:
            if usuarios_crudos != self.usuarios:
                raise ValueError(
                    f"{carpeta} y su sin_reduccion no tienen los mismos usuarios en el mismo "
                    "orden, asi que las etiquetas del modelo no se corresponden con los vectores.")
            # Contar usuarios no basta: un modelo de TF-IDF sobre los mismos
            # 598 pasa el filtro y luego da vecinos de otro espacio.
            dimensiones = np.load(carpeta / "X.npy").shape[1]
            if self.modelo.X.shape[1] != dimensiones:
                raise ValueError(
                    f"El modelo tiene {self.modelo.X.shape[1]} dimensiones y {carpeta} tiene "
                    f"{dimensiones}: no se entreno con esta representacion.")
        if len(self.usuarios) != self.modelo.X.shape[0]:
            raise ValueError(
                f"{carpeta_representacion} tiene {len(self.usuarios)} usuarios y el modelo se entreno con "
                f"{self.modelo.X.shape[0]}: no son la misma representacion (¿datos_1_a_600 frente a datos_completos?). "
                "Entrena el modelo con esa carpeta o pasa la carpeta con la que se entreno.")
        self.preprocesador = preprocesador
        self.textos = self._cargar_textos(csv_textos) if csv_textos else {}

    @staticmethod
    def _cargar_textos(ruta_csv) -> dict[str, str]:
        """Texto de cada usuario, recortado, para enseñar a los vecinos.

        Pre:  el CSV tiene subject_id y una columna text o text_Clustering.
        Post: {usuario: sus textos unidos, hasta 2000 caracteres}.
        """
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

        Pre:  hay preprocesador, o sea que venimos de la rama TF-IDF.
        Post: frase explicando por que no quedan tokens.
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
        """Texto crudo -> vector en el mismo espacio que el entrenamiento.

        Con la representacion densa el espacio es el de Roberta (1024), 
        la asignacion se hace por vecinos, no con modelo.predecir.

        Pre:  texto no vacio.
        Post: (vector (1, d), tokens). Lanza ValueError si el texto se queda
              sin nada que vectorizar.
        """
        if self.densa:
            # Roberta quiere el texto natural: al entrenar tampoco se lematizo.
            return self.embebedor(texto), texto.split()

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

        Pre:  el modelo lleva su X de entrenamiento.
        Post: (pca ajustado, coords (n, 2)). Se calcula una vez y se reutiliza.
        """
        if not hasattr(self, "_pca"):
            from sklearn.decomposition import PCA

            A = self.modelo.X.toarray() if sparse.issparse(self.modelo.X) else np.asarray(self.modelo.X)
            self._pca = PCA(n_components=2, random_state=semilla)
            self._coords = self._pca.fit_transform(A)
        return self._pca, self._coords

    def grafico(self, resultado: "Resultado", ruta) -> Path:
        """Guarda un PNG con la nube de usuarios proyectada a 2D por PCA.

        Pre:  resultado viene de consultar() con este mismo clasificador.
        Post: el PNG escrito, y su ruta. Crea las carpetas que falten.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        import colorsys

        from Clustering_jerarquico.algoritmo import distancias

        pca, coords = self._proyeccion()
        vector = (resultado.vector.toarray() if sparse.issparse(resultado.vector)
                  else np.asarray(resultado.vector))

        if self.densa:
            # El PCA esta hecho sobre las coordenadas de UMAP y el texto vive
            # en el de Roberta, asi que va a la media de sus vecinos.
            D = distancias(vector, self.embeddings, "cosine")[0]
            cercanos = np.argsort(D)[:10]
            peso = 1.0 / (D[cercanos] + 1e-9)
            punto = (coords[cercanos] * peso[:, None]).sum(0) / peso.sum()
        else:
            punto = pca.transform(vector)[0]
            D = distancias(resultado.vector, self.modelo.X, self.modelo.distancia)[0]
        etiquetas = self.modelo.etiquetas
        fila = {u: i for i, u in enumerate(self.usuarios)}
        cercanos = [fila[v.usuario] for v in resultado.vecinos if v.usuario in fila]

        # El tono dice de que cluster es y la saturacion su distancia real: la
        # posicion en 2D sola no vale para juzgar cercania.
        medias = {c: float(D[etiquetas == c].mean()) for c in range(self.modelo.k)}
        cerca, lejos = min(medias.values()), max(medias.values())
        margen = max(lejos - cerca, 1e-9)

        # Intensidad por usuario y no por grupo: dentro de un cluster hay
        # gente mucho mas parecida que otra.
        suyo = D.min(), D.max()
        punto_p = (1 - (D - suyo[0]) / max(suyo[1] - suyo[0], 1e-9)) ** 0.8

        orden = sorted(medias, key=medias.get)
        orden.remove(resultado.cluster)

        # Con Roberta todo cabe en un cono de 0.015, asi que con dos decimales
        # todos los grupos pondrian 0.01.
        separacion = max(lejos - cerca, 1e-9)
        decimales = min(6, max(2, int(np.ceil(-np.log10(separacion))) + 1))

        def pintar(c, tono, fuerza, etiqueta, asignado=False):
            miembros = np.flatnonzero(etiquetas == c)
            # Se reescala dentro del grupo para que el degradado se note aunque
            # todos sus miembros esten lejos del texto.
            p = punto_p[miembros]
            p = (p - p.min()) / max(p.max() - p.min(), 1e-9) if len(p) > 1 else np.ones_like(p)

            if asignado:
                # Al cuadrado: en un grupo grande casi todos caen en mitad
                # del rango y se verian iguales.
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
            # Distancia y no "1 - distancia" en porcentaje: tras el SVD hay
            # componentes negativas, la distancia pasa de 1 y salia negativo.
            pintar(c, tono, fuerza, f"cluster {c} — dist {medias[c]:.{decimales}f}")

        pintar(resultado.cluster, 0.0, 1.0,
               f"cluster {resultado.cluster} — ASIGNADO, dist {medias[resultado.cluster]:.{decimales}f}",
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

    def _cluster_por_votacion(self, X, cuantos: int = 15):
        """Respaldo sin el reductor: k-NN sobre las etiquetas del clustering.

        Si el texto no puede llegar al espacio de UMAP, se mira a quien se
        parece en el de Roberta y hereda su grupo.

        Pre:  estamos en la rama densa, o sea que hay embeddings cargados.
        Post: (cluster, (votos a favor, votos contados)).
        """
        from Clustering_jerarquico.algoritmo import distancias

        D = distancias(X, self.embeddings, "cosine")[0]
        cercanos = np.argsort(D)[:min(cuantos, len(D))]
        votos = collections.Counter(int(self.modelo.etiquetas[i]) for i in cercanos)
        cluster, a_favor = votos.most_common(1)[0]
        return cluster, (a_favor, len(cercanos))

    def consultar(self, texto: str, vecinos: int = 5) -> Resultado:
        """Asigna el texto a un cluster y busca los usuarios mas cercanos dentro.

        Pre:  texto no vacio y vecinos >= 1.
        Post: Resultado con el cluster, hasta `vecinos` usuarios ordenados de
              mas a menos parecido, y la ruta que se uso para asignarlo. Si el
              reductor falla al transformar, se pasa a la ruta de vecinos.
        """
        X, tokens = self.vectorizar(texto)

        from Clustering_jerarquico.algoritmo import distancias

        votos, ruta = None, ""
        if self.densa and self.reductor is not None:
            try:
                # El texto recorre la misma cadena que los datos de entrenamiento.
                reducido = self.reductor.transform(X)
            except Exception as error:       # por si numba falla aquí aun recompilando las distancias
                self.reductor = None
                self.sin_reductor = f"{type(error).__name__} al transformar"
        if self.densa and self.reductor is not None:
            ruta = "umap"
            cluster = int(self.modelo.predecir(reducido)[0])
            referencia, metrica = self.embeddings, "cosine"
        elif self.densa:
            ruta = "vecinos"
            cluster, votos = self._cluster_por_votacion(X)
            referencia, metrica = self.embeddings, "cosine"
        else:
            cluster = int(self.modelo.predecir(X)[0])
            referencia, metrica = self.modelo.X, self.modelo.distancia

        dentro = np.flatnonzero(self.modelo.etiquetas == cluster)
        D = distancias(X, referencia[dentro], metrica)[0]
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
            votos=votos,
            ruta=ruta,
        )
