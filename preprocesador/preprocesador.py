import html
import json
import os
import re
import xml.etree.ElementTree as eT
from functools import lru_cache
from contextlib import redirect_stdout

from langdetect import detect, LangDetectException
import matplotlib.pyplot as plt
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
    """
    Clase encargada de la carga, limpieza, normalización y lematización de textos
    provenientes de archivos XML para clustering.
    """
    def __init__(self, args, chat_words_path=Path(__file__).resolve().parent / "chat_words.json"):
        """
        Constructor de la clase Preprocesador.

        Precondiciones:
            - args debe ser un diccionario con los parámetros de configuración ('idioma', 'min_tokens', 'max_tokens'...).
            - chat_words_path debe ser una ruta válida (o inexistente manejada con seguridad) hacia un JSON de abreviaturas.

        Postcondiciones:
            - Se inicializan los recursos de NLTK, las estructuras de stopwords, los lematizadores y los atributos de configuración.
        """
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
        """
        Verifica y descarga los recursos necesarios de NLTK si no se encuentran disponibles.

        Precondiciones:
            - Conexión a internet disponible en caso de requerir descarga.

        Postcondiciones:
            - Los paquetes requeridos de NLTK quedan accesibles en el entorno local.
        """
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
        """
        Carga las abreviaturas informales desde un archivo JSON.

        Precondiciones:
            - json_path es una cadena o Path que apunta a un fichero JSON con pares clave-valor de abreviaturas.

        Postcondiciones:
            - Retorna un diccionario con las abreviaturas en minúsculas de una sola palabra, excluyendo stopwords ambiguas.
        """
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
        """
        Lee y procesa todos los archivos XML contenidos en el directorio indicado.

        Precondiciones:
            - directorio debe ser una ruta válida que contenga archivos XML con la estructura esperada (sujeto, ID, WRITING, TITLE, DATE, TEXT, INFO),
              o el CSV que guarda --guardar_csv (output/data.csv), con esas mismas columnas.
            - Hay al menos una publicación.

        Postcondiciones:
            - Retorna un DataFrame de Pandas con las columnas: subject_id, title, date, text, info, con fechas parseadas.
        """
        dir_data = Path(directorio)
        if dir_data.suffix.lower() == ".csv":
            # Todo como texto y sin convertir vacíos a NaN: así queda igual que leyendo los XML
            df_data = pd.read_csv(dir_data, dtype=str, keep_default_na=False)
            df_data["date"] = pd.to_datetime(df_data["date"], errors='coerce')
            return df_data

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

        if not publicaciones:
            raise ValueError(f"No hay publicaciones en ningún XML de {dir_data}")
        df_data = pd.DataFrame(publicaciones)
        df_data["date"] = pd.to_datetime(df_data["date"], errors='coerce')

        return df_data

    @staticmethod
    def _texto_crudo(df):
        """
        Concatena el título y el texto de cada registro en el DataFrame.

        Precondiciones:
            - df es un DataFrame que contiene las columnas 'title' y 'text'.

        Postcondiciones:
            - Retorna una Serie de Pandas con los textos combinados y limpios de espacios extremos.
        """
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
        """
        Elimina etiquetas y entidades HTML de una cadena de texto.

        Precondiciones:
            - text es de tipo str.

        Postcondiciones:
            - Retorna el texto sin etiquetas HTML ni entidades codificadas.
        """
        text = html.unescape(text) # para entidades HTML: &lt...
        return RE_HTML.sub(" ", text)

    @staticmethod
    def _remove_url(text):
        """
        Elimina URLs de una cadena de texto.

        Precondiciones:
            - text es de tipo str.

        Postcondiciones:
            - Retorna el texto sin URLs detectadas.
        """
        return RE_URL.sub(" ", text)

    @staticmethod
    def _chat_conversion(tokens, chat_words):
        """
        Reemplaza abreviaturas de chat por su forma extendida en una lista de tokens.

        Precondiciones:
            - tokens es una lista de strings.
            - chat_words es un diccionario de mapeo de abreviaturas.

        Postcondiciones:
            - Retorna la lista de tokens con las abreviaturas reemplazadas.
        """
        return [chat_words.get(t, t) for t in tokens]

    @staticmethod
    def _negaciones_en(text):
        """
        Normaliza las contracciones de negación en inglés.

        Precondiciones:
            - text es de tipo str.

        Postcondiciones:
            - Retorna el texto con las negaciones expandidas (can't -> can not) y apóstrofes auxiliares eliminados.
        """
        text = re.sub(r"\bcan(?:'t|not)\b", "can not", text) # can't, cannot
        text = re.sub(r"\bwon't\b", "will not", text) # won't
        text = re.sub(r"\bain't\b", "is not", text) # ain't
        text = re.sub(r"(?<=[a-z])n't\b", " not", text) # isn't, don't... el resto
        return re.sub(r"(?<=[a-z])'(?:s|m|re|ve|ll|d)\b", "", text) # 's, 'm... y las quita

    # endregion

    # region Métodos de NORMALIZACIÓN
    def _normalize_text(self, text):
        """
        Normaliza el texto convirtiéndolo a minúsculas, unificando apóstrofes y reduciendo alargamientos de letras.

        Precondiciones:
            - text es de tipo str.

        Postcondiciones:
            - Retorna el texto normalizado.
        """
        text = text.lower().translate(APOSTROFES)  # minúsculas y apóstrofos curvos -> rectos
        if "'" in text:
            text = self._negaciones_en(text)  # (en español no afecta)
        return RE_ALARGA.sub(r"\1\1", text)  # soooo -> soo

    def _detectar_idioma(self, tokens):
        """
        Detecta si el idioma predominante de los tokens es español o inglés.

        Precondiciones:
            - tokens es una lista de strings.

        Postcondiciones:
            - Retorna 'spanish' o 'english' según la frecuencia de palabras exclusivas.
        """
        muestra = tokens[:60]
        en = sum(t in self.solo_en for t in muestra)
        es = sum(t in self.solo_es for t in muestra)
        return "spanish" if es > en else "english"

    # endregion

    # region Métodos de LEMATIZACIÓN
    def _lemma_en(self, palabra):
        """
        Lematiza una palabra en inglés probando primero como sustantivo y luego como verbo.

        Precondiciones:
            - palabra es de tipo str.

        Postcondiciones:
            - Retorna el lema correspondiente en inglés.
        """
        lema = self.lemmatizer.lemmatize(palabra, "n")
        return lema if lema != palabra else self.lemmatizer.lemmatize(palabra, "v")

    def _lemmatize(self, doc, language):
        """
        Aplica lematización (inglés) o stemming (español) a una lista de tokens según el idioma.

        Precondiciones:
            - doc es una lista de strings.
            - language es un string ('english' o 'spanish').

        Postcondiciones:
            - Retorna una lista de tokens lematizados/raíces.
        """
        if language == "english":
            return [self._lema(t) for t in doc]
        return [self._raiz(t) for t in doc]  # español: stemming (NLTK no lematiza español)
    #endregion

    # region Pipeline General
    def _procesar_texto(self, doc):
        """
        Ejecuta el pipeline completo de limpieza, filtrado y lematización sobre un texto individual.

        Precondiciones:
            - doc es una cadena de texto (str).

        Postcondiciones:
            - Retorna una lista de tokens procesados o None si el texto no cumple con los criterios de longitud mínima.
        """
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
        """
        Procesa de forma optimizada los textos únicos de un DataFrame mediante el pipeline de texto.

        Precondiciones:
            - data es un DataFrame que contiene la columna 'texto_crudo'.

        Postcondiciones:
            - Retorna el DataFrame con una nueva columna 'text_Clustering' con los tokens procesados.
        """
        unicos = data["texto_crudo"].unique()
        resultados = {}

        for t in tqdm(unicos, desc="Preprocesando textos"):
            resultados[t] = self._procesar_texto(t)

        data["text_Clustering"] = [resultados[t] for t in data["texto_crudo"]]
        return data

    def preprocesar_datos_clustering(self):
        """
        Ejecuta el flujo completo de preprocesamiento de datos para clustering, incluyendo carga, limpieza y exportación opcional.

        Precondiciones:
            - self.args debe contener la clave 'datos' con la ruta al directorio de entrada.

        Postcondiciones:
            - Actualiza y retorna el DataFrame procesado (self.df), eliminando vacíos y duplicados, y opcionalmente guarda los resultados en CSV.
        """
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

    # region Métodos de Análisis de Datos

    def _codigo_idioma(self, texto):
        """
        Detecta el idioma de un texto y devuelve su código ISO correspondiente.

        Precondiciones:
            - texto es de tipo str.
            - Las bibliotecas y funciones de limpieza (como detect, RE_USUARIO, etc.) se encuentran disponibles.

        Postcondiciones:
            - Retorna el código ISO del idioma detectado o 'und' en caso de que ocurra una excepción durante la detección.
        """
        texto = RE_USUARIO.sub(" ", self._remove_url(self._remove_html_tags(texto)))[:500]
        try:
            return detect(texto)
        except LangDetectException:
            return "und"

    def analisis_exploratorio(self, guardar_fig=None, guardar_txt="output/analisis.txt"):
        """
        Ejecuta el análisis exploratorio de los datos y redirige la salida estándar para escribir los resultados en un archivo de texto.

        Precondiciones:
            - guardar_txt es una ruta de archivo válida donde se escribirán los resultados.

        Postcondiciones:
            - Se crea el directorio de destino si no existe, se guarda el informe del análisis en el archivo de texto y se retorna el resultado del análisis.
        """
        Path(guardar_txt).parent.mkdir(parents=True, exist_ok=True)
        with open(guardar_txt, "w", encoding="utf-8") as f, redirect_stdout(f):
            return self._analizar(guardar_fig)

    def _analizar(self, guardar_fig=None):
        """
        Realiza un resumen detallado de calidad, longitudes, duplicados e idiomas del dataset, generando opcionalmente un gráfico.

        Precondiciones:
            - self.df_original o self.args['datos'] deben estar disponibles para cargar los datos en caso de ser necesario.

        Postcondiciones:
            - Imprime métricas de conteo, valores ausentes, estadísticas de longitud y distribución de idiomas, muestra o guarda el gráfico resultante y retorna el conteo de idiomas.
        """
        if self.df_original is None:
            self.df_original = self._cargar_data(self.args["datos"])

        df = self.df_original

        texto = self._texto_crudo(df).fillna("").astype(str)
        texto = texto[texto.str.strip().ne("")]

        # Resumen y valores ausentes
        print(f"Publicaciones: {len(df)} | Sujetos: {df['subject_id'].nunique()} | "f"Fechas: {df['date'].min()} -> {df['date'].max()}")
        vacios = (df.isna() | df.astype(str).apply(lambda col: col.str.strip().eq(""))).sum()
        print("\nAusentes/vacíos por columna:")
        print(pd.DataFrame({"n": vacios, "%": (100 * vacios / len(df)).round(2)}))

        # Longitudes y duplicados
        if texto.empty:
            print("\nNo hay textos no vacíos para calcular longitudes.")
        else:
            longitud = pd.DataFrame({
                "caracteres": texto.str.len(),
                "palabras": texto.str.split().str.len()
            })
            print("\nLongitud de las publicaciones:")
            print(longitud.describe(percentiles=[.05, .5, .95]).round(1).loc[["min", "5%", "50%", "mean", "95%", "max"]])

        print(f"\nTextos duplicados exactos: {texto.duplicated().sum()}")

        # Idiomas: detectar solo una vez por texto único
        mapa = {
            t: self._codigo_idioma(t)
            for t in tqdm(texto.unique(), desc="Detectando idioma")
        }
        codigos = texto.map(mapa)
        categorias = codigos.map({"en": "Inglés", "es": "Español"}).fillna("Otro")
        orden = ["Inglés", "Español", "Otro"]
        conteo = categorias.value_counts().reindex(orden, fill_value=0)
        pct = conteo.div(conteo.sum()).mul(100).fillna(0)

        print("\nDistribución de idiomas:")
        print(pd.DataFrame({"n": conteo, "%": pct.round(2)}))
        print("\nCódigos incluidos en 'Otro' (und = no detectado):")
        print(codigos[categorias.eq("Otro")].value_counts().head(5).to_string())

        # Gráfico
        fig, ax = plt.subplots(figsize=(6, 4))
        barras = ax.bar(conteo.index, conteo.values, color=["blue", "orange", "gray"])
        ax.bar_label(barras, labels=[f"{n}\n({p:.1f}%)" for n, p in zip(conteo, pct)])
        ax.set(title="Distribución de idiomas", ylabel="Publicaciones")
        ax.margins(y=0.15)
        fig.tight_layout()

        if guardar_fig:
            Path(guardar_fig).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(guardar_fig, dpi=150)

        plt.show()
        plt.close(fig)

        return conteo

    # endregion
