import html
import json
import os
import re
import xml.etree.ElementTree as eT
from functools import lru_cache

import pandas as pd
from pathlib import Path
from tqdm import tqdm

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer

# Expresiones regulares: se compilan UNA vez aquí (no en cada texto)
RE_HTML = re.compile(r"</?[A-Za-z][^>]{0,200}>")                        # <b>, </a>...
RE_URL = re.compile(r"(?:https?://|www\.)\S+", re.I)
RE_USUARIO = re.compile(r"(?<!\w)(?:/?u/|@)[\w-]+")                     # &#x200B; de Reddit
RE_ALARGA = re.compile(r"([^\W\d_])\1{2,}")                             # soooo -> soo
RE_PALABRA = re.compile(r"[^\W_]+(?:-[^\W_]+)*")                        # letras (con tildes) y cifras
APOSTROFES = str.maketrans({"’": "'", "‘": "'", "´": "'", "`": "'"})

class Preprocesador:
    def __init__(self, args, chat_words_path="./chat_words.json"):
        self._descargar_nltk()
        self.args = args
        self.chat_words = self._cargar_chat_words(chat_words_path)

        self.idioma = args.get("idioma", "auto")  # "auto", "english" o "spanish"
        self.min_tokens = args.get("min_tokens", 3)  # descarta publicaciones más cortas
        self.max_tokens = args.get("max_tokens", 1000)  # trunca las más largas

        # Stopwords (sin negaciones)
        negaciones = {"no", "not", "nor", "ni", "sin"}
        en = {w.replace("'", "") for w in stopwords.words("english")}
        es = set(stopwords.words("spanish"))
        self.stopwords = {"english": en - negaciones, "spanish": es - negaciones}
        self.solo_en, self.solo_es = en - es, es - en  # para adivinar el idioma en modo auto

        # Lematizador (EN) y Stemmer (ES). Cache para que calcule una vez cada palabra
        self.lemmatizer = WordNetLemmatizer()
        self._lema = lru_cache(maxsize=None)(self._lemma_en)
        self._raiz = lru_cache(maxsize=None)(SnowballStemmer("spanish").stem)

        self.df_original = None
        self.df = None

    # region Métodos de CARGA
    @staticmethod
    def _descargar_nltk():
        nltks = [
            ('averaged_perceptron_tagger_eng', 'taggers/averaged_perceptron_tagger_eng'),
            ('punkt_tab', 'tokenizers/punkt_tab'),
            ('wordnet', 'corpora/wordnet'),
            ('stopwords', 'corpora/stopwords')
        ]
        for recurso, ruta in nltks:
            try:
                nltk.data.find(ruta)
            except LookupError:
                nltk.download(recurso, quiet=True, raise_on_error=True)

    @staticmethod
    def _cargar_chat_words(json_path):
        """Carga las abreviaturas si son solo de 1 palabra y no se consideran stopwords"""
        path = Path(json_path)
        if not path.exists():
            print(f"No se ha encontrado {json_path}. Se usará un diccionario vacío.")
            return {}
        with open(path, "r", encoding="utf-8") as f:
            datos = json.load(f)
        ambiguas = set(stopwords.words("english"))
        return {
            k.lower(): v.lower() for k, v in datos.items()
            if len(v.split()) == 1 and k.lower() not in ambiguas
        }

    @staticmethod
    def _cargar_data(directorio):
        dir_data = Path(directorio)
        publicaciones = []

        # Leemos todos los xml
        for path_xml in dir_data.rglob("*.xml"):
            sujeto = eT.parse(path_xml).getroot()

            # Obtenemos los metadatos
            subject_id = sujeto.findtext("ID")
            for writing in sujeto.findall(".//WRITING"):
                publicaciones.append({
                    "subject_id": subject_id,
                    "title": writing.findtext("TITLE", default=""),
                    "date": writing.findtext("DATE", default=""),
                    "text": writing.findtext("TEXT", default=""),
                    "info": writing.findtext("INFO", default="")
                })

        df_data = pd.DataFrame(publicaciones)
        df_data["date"] = pd.to_datetime(df_data["date"], errors='coerce')

        return df_data

    @staticmethod
    def _texto_crudo(df):
        # Obtener título y texto
        titulo = df["title"].fillna("").str.strip()
        texto = df["text"].fillna("").str.strip()

        # Unir título y texto
        resultado = titulo + " " + texto

        return resultado.str.strip()
    #endregion

    # region Métodos de LIMPIEZA-TRANSFORMACIÓN
    @staticmethod
    def _remove_html_tags(text):
        text = html.unescape(text) # para entidades HTML: &lt...
        return RE_HTML.sub(" ", text)

    @staticmethod
    def _remove_url(text):
        return RE_URL.sub(" ", text)

    @staticmethod
    def _chat_conversion(tokens, chat_words):
        return [chat_words.get(t, t) for t in tokens]

    @staticmethod
    def _negaciones_en(text):
        """Negaciones a not y fuera el resto de sufijos"""
        text = re.sub(r"\bcan(?:'t|not)\b", "can not", text) # can't, cannot
        text = re.sub(r"\bwon't\b", "will not", text) # won't
        text = re.sub(r"\bain't\b", "is not", text) # ain't
        text = re.sub(r"(?<=[a-z])n't\b", " not", text) # isn't, don't... el resto
        return re.sub(r"(?<=[a-z])'(?:s|m|re|ve|ll|d)\b", "", text) # 's, 'm... y las quita

    # endregion

    # region Métodos de NORMALIZACIÓN
    def _normalize_text(self, text):
        """Minúsculas, apóstrofes a mismo formato, negaciones (EN) """
        text = text.lower().translate(APOSTROFES)  # minúsculas y apóstrofos curvos -> rectos
        if "'" in text:
            text = self._negaciones_en(text)  # (en español no afecta)
        return RE_ALARGA.sub(r"\1\1", text)  # soooo -> soo

    def _detectar_idioma(self, tokens):
        """Cuenta palabras de cada idioma"""
        muestra = tokens[:60]
        en = sum(t in self.solo_en for t in muestra)
        es = sum(t in self.solo_es for t in muestra)
        return "spanish" if es > en else "english"

    # endregion

    # region Métodos de LEMATIZACIÓN
    def _lemma_en(self, palabra):
        """Lematiza como sustantivo y si no cambia, como verbo"""
        lema = self.lemmatizer.lemmatize(palabra, "n")
        return lema if lema != palabra else self.lemmatizer.lemmatize(palabra, "v")

    def _lemmatize(self, doc, language):
        if language == "english":
            return [self._lema(t) for t in doc]
        return [self._raiz(t) for t in doc]  # español: stemming (NLTK no lematiza español)
    #endregion

    # region Pipeline General
    def _procesar_texto(self, doc):
        """Devuelve la lista de tokens, o None si se descarta."""
        doc = doc[: self.max_tokens * 20]  # Recorte para textos muy largos

        # 1. Limpiar HTML, URL y menciones
        doc = self._remove_html_tags(doc)
        doc = self._remove_url(doc)
        doc = RE_USUARIO.sub(" ", doc)

        # 2. Normalizar y tokenizar (regex Unicode: quita también ¿ ¡ « » “ ”)
        doc = self._normalize_text(doc)
        tokens = RE_PALABRA.findall(doc)
        if not tokens:
            return None

        # 3. Idioma
        lan = self._detectar_idioma(tokens) if self.idioma == "auto" else self.idioma

        # 4. Abreviaturas (solo inglés)
        if lan == "english":
            tokens = self._chat_conversion(tokens, self.chat_words)

        # 5. Quitar números, palabras de 1 letra o >25, y stopwords
        sw = self.stopwords[lan]
        tokens = [t for t in tokens if t.replace("-", "").isalpha() and 2 <= len(t) <= 25 and t not in sw]
        # Acotar los límites de tokens establecido
        if len(tokens) < self.min_tokens:
            return None
        tokens = tokens[: self.max_tokens]

        # 6. Lematizar
        return self._lemmatize(tokens, lan)

    def _pipeline_limpieza_texto(self, data):
        """Los textos distintos se procesan una vez y a partir del diccionario se actualizan todos"""
        unicos = data["texto_crudo"].unique()
        resultados = {}

        for t in tqdm(unicos, desc="Preprocesando textos"):
            resultados[t] = self._procesar_texto(t)

        data["text_Clustering"] = [resultados[t] for t in data["texto_crudo"]]
        return data

    def preprocesar_datos_clustering(self):
        self.df_original = self._cargar_data(self.args["datos"])
        df = self.df_original.copy()

        # Texto útil, orden cronológico y duplicados exactos del mismo usuario
        df["texto_crudo"] = self._texto_crudo(df)
        df = df[df["texto_crudo"] != ""] # Eliminar vacíos
        df = df.sort_values(["subject_id", "date"], kind="stable")
        df = df.drop_duplicates(["subject_id", "texto_crudo"])

        df = self._pipeline_limpieza_texto(df)
        df = df[df["text_Clustering"].notna()].copy() # Eliminar vacíos

        # Duplicados tras limpiar
        clave = df["subject_id"] + "|" + df["text_Clustering"].map(" ".join)
        df = df[~clave.duplicated()]

        self.df = df.drop(columns="texto_crudo").reset_index(drop=True)
        perdidos = set(self.df_original["subject_id"]) - set(self.df["subject_id"])
        print(f"Aviso: {len(perdidos)} sujeto(s) sin texto útil.")

        if self.args.get("guardar_csv", False):
            os.makedirs("output", exist_ok=True)
            self.df_original.to_csv("output/data.csv", index=False)
            salida = self.df.copy()
            salida["text_Clustering"] = salida["text_Clustering"].map(" ".join)
            salida.to_csv("output/data_preprocesado.csv", index=False)

        return self.df

    #endregion
