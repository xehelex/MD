import json
import os
import re
import string
import warnings
import xml.etree.ElementTree as eT

from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from langdetect import detect, LangDetectException
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer

class Preprocesador:
    def __init__(self, args, chat_words_path="./config/chat_words.json"):
        self._descargar_nltk()
        self.chat_words = self._cargar_chat_words(chat_words_path)

        self.stopwords = {
            "english": set(stopwords.words("english")),
            "spanish": set(stopwords.words("spanish"))
        }
        self.lemmatizer = WordNetLemmatizer()

        self.df_original = None
        self.df = None

        self.args = args

    # region Métodos de CARGA-CONFIG
    @staticmethod
    def _descargar_nltk():
        nltks = ['averaged_perceptron_tagger_eng', 'punkt_tab', 'wordnet', 'stopwords']
        for recurso in nltks:
            nltk.download(recurso, quiet=True, raise_on_error=True)

    @staticmethod
    def _cargar_chat_words(json_path):
        path = Path(json_path)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            print(f"No se ha encontrado {json_path}. Se usará un diccionario vacío.")
            return {}

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
        if not df_data.empty:
            df_data["date"] = pd.to_datetime(df_data["date"], errors='coerce')

        if not os.path.exists('./output'):
            os.makedirs('./output')
        df_data.to_csv('output/data.csv', index=False)

        return df_data

    #endregion

    # region Métodos de LIMPIEZA-TRANSFORMACIÓN
    @staticmethod
    def _remove_html_tags(text):
        if not text.strip():
            return text
        with warnings.catch_warnings(record=False):
            warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
            return BeautifulSoup(text, "html.parser").get_text(separator=" ", strip=True)

    @staticmethod
    def _remove_url(text):
        return re.sub(r"https?://\S+|www\.\S+"," ", text, flags=re.IGNORECASE)

    @staticmethod
    def _remove_punc(text, punc):
        return text.translate(str.maketrans({char: " " for char in punc}))

    @staticmethod
    def _remove_stopwords(doc, stopword):
        return [
            token for token in doc
            if token not in stopword
        ]

    @staticmethod
    def _chat_conversion(text, chat_words):
        new_text = []
        for i in text.split():
            if i.upper() in chat_words:
                new_text.append(chat_words[i.upper()].lower())
            else:
                new_text.append(i)
        return " ".join(new_text)

    @staticmethod
    def _negaciones_en(text):
        text = re.sub(r"n't\b", "nt", text, flags=re.IGNORECASE)  # Pasar de n't a nt
        text = re.sub(r"['’]\w+", "", text, flags=re.IGNORECASE)  # El resto de sufijos fuera

        return text

    #endregion

    # region Métodos de NORMALIZACIÓN
    def _normalize_text(self, text, punc, language):
        text = text.lower() # Minúsculas

        if language == "english":
            text = self._negaciones_en(text) # Para no perder negaciones

        text = self._remove_punc(text, punc) # Signos de puntuación fuera

        text = re.sub(r"\s+", " ", text).strip() # Espacios y saltos de línea

        return text

    #endregion

    # region Métodos de LEMATIZACIÓN-POSTAGGING

    @staticmethod
    def _get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def _lemmatize(self, doc, language):
        if language != "english": # TODO de momento solo textos en inglés
            return doc

        pos_tagged = pos_tag(doc)

        clean_tokens = [
            self.lemmatizer.lemmatize(token, self._get_wordnet_pos(tag))
            for token, tag in pos_tagged
        ]
        return clean_tokens

    #endregion

    # region Pipeline General

    def _pipeline_limpieza_texto(self, data):
        columna_texto="text" #TODO con un arg
        docs_preprocesados = []
        punc = string.punctuation.replace("-", "").replace("'", "") # Signos de puntuación, menos apóstrofes y guiones

        for doc in tqdm(data[columna_texto], desc=f"Preprocesando textos"):

            # 1. Gestionar NaN
            if pd.isnull(doc) or not str(doc).strip():
                tokens = []
            else:
                doc = str(doc)
                try:
                    lan = 'spanish' if detect(doc) == "es" else "english"  # De momento solo EN y ES
                except LangDetectException:
                    lan = "english"  # Idioma por defecto

                # 2. Limpiar HTML y URL
                doc = self._remove_url(doc)  # Quitar URLs
                doc = self._remove_html_tags(doc)  # Quitar HTML

                # 3. Manejo de jerga (EN)
                doc = self._chat_conversion(doc, self.chat_words)  # Tratar jerga (abreviaturas)

                # 4. Normalizar
                doc = self._normalize_text(doc, punc, lan)

                # 5. Tokenizar
                tokens = word_tokenize(doc, language=lan)

                # 6. Stopwords
                tokens = self._remove_stopwords(tokens, self.stopwords[lan])

                # 7. Lematizar con POS tagging (identifica si es noun, verb...)
                tokens = self._lemmatize(tokens, lan)

            docs_preprocesados.append(tokens)

        columna_texto_modificada = columna_texto + "_Clustering"
        data[columna_texto_modificada] = docs_preprocesados

        return data

    def preprocesar_datos_clustering(self):
        #TODO cambiar lo del directorio por un arg
        self.df_original = self._cargar_data(self.args["data"])
        self.df = self.df_original.copy()
        self.df = self._pipeline_limpieza_texto(self.df)

        return self.df

    #endregion

# Para pruebas
if __name__ == '__main__':
    argus = {"data": "C:\\Users\\shama\\OneDrive\\Escritorio\\data_1_to_600\\auxiliar"} # TODO para pruebas IMPORTANTE CAMBIAR VUESTRA RUTA
    prepro = Preprocesador(argus)

    df = prepro.preprocesar_datos_clustering()
    df.to_csv('output/data_prepro.csv', index=False)