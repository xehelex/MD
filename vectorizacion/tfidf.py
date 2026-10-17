
from dataclasses import dataclass
from datetime import datetime
import html
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class Historial:
    """Identificador y publicaciones útiles, ordenadas cronológicamente."""

    usuario: str
    archivo: str
    publicaciones: tuple[str, ...]
    publicaciones_totales: int


def reparar_entidades(texto: str) -> str:
    """Reconstruye las entidades HTML que el corpus trae rotas.

    Los XML de CLEF llegan con "don #8217;t" donde deberia poner "don't": se
    perdio el & de &#8217; y ademas se colo un espacio. Son unas 43.000 palabras
    partidas en los 600 primeros sujetos, entre negaciones, posesivos y formas
    como I'm o I've. Si no se arregla aqui, el tokenizador produce basura del
    tipo "don" + "8217" + "t" y, lo que es peor, las negaciones se pierden.

    Exige el punto y coma final, asi que no toca "C#", "issue #123" ni "100 #5".
    """
    texto = re.sub(r"\s?#(\d{2,6});", r"&#\1;", texto)
    texto = re.sub(r"(?<![&\w])\s?(amp|quot|gt|lt|nbsp);", r"&\1;", texto)
    return html.unescape(texto)


def limpiar_texto(texto: str) -> str:
    """Recibe texto XML decodificado; devuelve texto sin URLs ni menciones reales.

    Conserva negaciones, cifras y puntuación; no requiere recursos de NLTK.
    El tokenizador posterior selecciona los términos para TF-IDF.
    """
    # Primero de todo: sin esto, las sustituciones de negaciones de abajo no
    # llegan a dispararse nunca, porque el texto real dice "don #8217;t".
    texto = reparar_entidades(texto)
    texto = re.sub(r"https?://\S+|www\.\S+", " URLTOKEN ", texto, flags=re.I)
    texto = re.sub(r"(?<!\w)(?:/?u/|@)[\w-]+", " USERTOKEN ", texto)
    # Evita que el tokenizador de palabras pierda la negación de can't/don't.
    texto = texto.replace("’", "'")
    texto = re.sub(r"\bcan't\b", "cannot", texto, flags=re.I)
    texto = re.sub(r"\bwon't\b", "will not", texto, flags=re.I)
    texto = re.sub(r"n't\b", " not", texto, flags=re.I)
    return " ".join(texto.split())


def cargar_historiales(carpeta: str | Path) -> list[Historial]:
    """Lee XML INDIVIDUAL/WRITING de una carpeta, sin modificar los originales.

    Requiere IDs únicos y fechas ISO válidas. Devuelve una instancia por XML.
    Excluye publicaciones sin título ni texto; falla si un usuario queda vacío.
    """
    archivos = sorted(Path(carpeta).glob("*.xml"))
    if not archivos:
        raise ValueError(f"No hay archivos XML en {carpeta}")
    resultado = []
    ids = set()
    for archivo in archivos:
        raiz = ET.parse(archivo).getroot()
        usuario = (raiz.findtext("ID") or "").strip()
        if raiz.tag != "INDIVIDUAL" or not usuario or usuario in ids:
            raise ValueError(f"Raíz o ID ausente/duplicado: {archivo.name}")
        ids.add(usuario)
        publicaciones = raiz.findall("WRITING")
        ordenadas = sorted(
            publicaciones,
            key=lambda p: datetime.fromisoformat(p.findtext("DATE") or ""),
        )
        textos = []
        for publicacion in ordenadas:
            partes = []
            for campo in ("TITLE", "TEXT"):
                valor = (publicacion.findtext(campo) or "").strip()
                if valor.lower() not in ("", "[deleted]", "[removed]"):
                    partes.append(valor)
            texto = limpiar_texto(". ".join(partes))
            if texto:
                textos.append(texto)
        if not textos:
            raise ValueError(f"Usuario sin texto útil: {usuario}")
        resultado.append(Historial(usuario, archivo.name, tuple(textos), len(publicaciones)))
    return resultado


class TfidfPorUsuario(TfidfVectorizer):
    """TF-IDF con una secuencia de publicaciones por documento/usuario.

    Conserva fit_transform/transform de sklearn. Los recuentos se agregan por
    usuario, pero los n-gramas nunca cruzan el límite entre publicaciones.
    Admite análisis por palabras o caracteres, siempre dentro de cada publicación.
    """

    def build_analyzer(self):
        analizar_texto = super().build_analyzer()

        def analizar_usuario(publicaciones):
            if isinstance(publicaciones, str):
                raise TypeError("Se esperaba una secuencia de publicaciones, no un string.")
            for publicacion in publicaciones:
                yield from analizar_texto(publicacion)

        return analizar_usuario


# Palabras que NO se eliminan aunque esten en la lista de vacias. Sin ellas
# "no quiero morir" y "quiero morir" llegan identicas al clustering, que para
# esta tarea es justo la distincion que importa.
NEGACIONES = frozenset({
    "not", "no", "never", "none", "nobody", "nothing", "nowhere",
    "neither", "nor", "cannot", "without", "against",
})


def palabras_vacias_sin_negaciones() -> list[str]:
    """Lista de stopwords inglesas de NLTK, menos las que niegan.

    Se importa aqui dentro y no arriba para que el modulo siga funcionando sin
    NLTK instalado mientras no se pidan stopwords. La resta de conjuntos no
    necesita saber que hay en la lista: lo que no este, no se quita.
    """
    from nltk.corpus import stopwords

    try:
        vacias = set(stopwords.words("english"))
    except LookupError:  # el corpus no esta descargado todavia
        import nltk

        nltk.download("stopwords", quiet=True, raise_on_error=True)
        vacias = set(stopwords.words("english"))

    return sorted(vacias - NEGACIONES)


def crear_vectorizador(
    ngramas: int = 1,
    min_df: int = 3,
    max_df: float = 1.0,
    max_features: int | None = None,
    sublinear_tf: bool = False,
    representacion: str = "palabras",
    quitar_vacias: bool = False,
) -> TfidfPorUsuario:
    """Crea un vectorizador sin ajustar, con los valores base de SAD_Plantilla.

    Requiere ngramas/min_df positivos y 0 < max_df <= 1. Devuelve matrices CSR
    float32, normalizadas L2; vocabulario e IDF se aprenden al llamar a fit.
    'palabras' usa longitudes 1..ngramas; 'caracteres4' usa exactamente cuatro
    caracteres (incluidos espacios y puntuación), sin padding de palabras.
    """
    if ngramas < 1 or min_df < 1 or not 0 < max_df <= 1:
        raise ValueError("ngramas/min_df deben ser positivos y max_df estar en (0, 1].")
    if max_features is not None and max_features < 1:
        raise ValueError("max_features debe ser positivo.")
    if representacion not in ("palabras", "caracteres4"):
        raise ValueError("representacion debe ser 'palabras' o 'caracteres4'.")
    if representacion == "caracteres4" and ngramas != 1:
        raise ValueError("ngramas solo configura palabras; caracteres4 tiene longitud fija 4.")
    if quitar_vacias and representacion == "caracteres4":
        raise ValueError("Las palabras vacias no se pueden filtrar sobre n-gramas de caracteres.")
    return TfidfPorUsuario(
        analyzer="char" if representacion == "caracteres4" else "word",
        ngram_range=(4, 4) if representacion == "caracteres4" else (1, ngramas),
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
        norm="l2",
        stop_words=palabras_vacias_sin_negaciones() if quitar_vacias else None,
        dtype=np.float32,
    )
