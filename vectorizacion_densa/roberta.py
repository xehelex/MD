import html
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Expresiones regulares (se compilan una sola vez)
RE_URL = re.compile(r"(?:https?://|www\.)\S+", re.I)
RE_USUARIO = re.compile(r"(?<!\w)(?:/?u/|@)[\w-]+")
RE_ESPACIOS = re.compile(r"[\s\u200b]+")

class RobertaPorUsuario:
    """
    Genera un embedding por post con RoBERTa y los agrega por usuario (media).
    """

    def __init__(self,
        data_path="output/data.csv",
        model_name="FacebookAI/roberta-large",
        batch_size=32,
        max_length=512,
        preprocess=True,
        normalizar = False
    ):
        """
        Constructor de la clase RobertaPorUsuario.

        Precondiciones:
            - data_path es una ruta válida al archivo CSV de entrada.
            - model_name corresponde a un identificador válido de modelo en Hugging Face.

        Postcondiciones:
            - Se configuran los parámetros de ejecución, se detecta el dispositivo disponible (GPU/CPU) y se cargan el tokenizador y el modelo correspondiente.
        """
        self.data_path = Path(data_path)
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.preprocess = preprocess
        self.normalizar = normalizar

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Dispositivo: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

        # En GPU, media precisión: ~2x más rápido y la mitad de memoria
        if self.device.type == "cuda":
            self.model.half()

        # No entrenamos: solo extraemos embeddings
        self.model.eval()

        self.post_embeddings = None
        self.user_embeddings = None

    # region Datos + Prepro-mini

    def clean_text(self, text):
        """
        Realiza una limpieza ligera del texto sin eliminar stopwords, negaciones ni aplicar lematización.

        Precondiciones:
            - text es de tipo str.

        Postcondiciones:
            - Retorna el texto depurado de espacios sobrantes y opcionalmente libre de URLs, etiquetas HTML y menciones si está activado el preprocesamiento.
        """
        text = text.strip()

        if self.preprocess:
            text = html.unescape(text)       # &amp; &#x200B; ...
            text = RE_URL.sub(" ", text)     # las URLs gastan muchos tokens
            text = RE_USUARIO.sub(" ", text)
            text = RE_ESPACIOS.sub(" ", text).strip()

        return text

    def load_data(self):
        """
        Lee el archivo CSV de datos y extrae un DataFrame filtrado con los identificadores de usuario y textos útiles.

        Precondiciones:
            - self.data_path apunta a un archivo CSV accesible con las columnas requeridas (subject_id, title, text).

        Postcondiciones:
            - Retorna un DataFrame indexado nuevamente que contiene únicamente registros con texto y sujetos válidos. Lanza un ValueError si el resultado está vacío.
        """

        # Solo las columnas necesarias; los vacíos quedan como "" (no NaN)
        df = pd.read_csv(
            str(self.data_path),
            usecols=["subject_id", "title", "text"],
            dtype=str,
            keep_default_na=False,
        ) # type: ignore

        total = len(df)

        # Título + texto
        texto = df["title"] + " " + df["text"]

        df = pd.DataFrame({
            "subject_id": df["subject_id"].str.strip(),
            "text": texto.map(self.clean_text),
        })
        df = df[(df["subject_id"] != "") & (df["text"] != "")]

        print(f"Filas en el CSV: {total} | con texto utilizable: {len(df)}")

        if df.empty:
            raise ValueError("No queda ningún post con texto.")

        return df.reset_index(drop=True)

    #endregion

    # region Vectorización

    @torch.inference_mode()
    def _encode_chunk(self, texts, pbar):
        """
        Calcula la media de los embeddings de los tokens de cada lote de textos ignorando el padding.

        Precondiciones:
            - texts es una lista de cadenas de texto.
            - pbar es una instancia activa de tqdm para la barra de progreso.

        Postcondiciones:
            - Retorna un arreglo de numpy con los embeddings codificados y opcionalmente normalizados.
        """
        resultados = []

        for i in range(0, len(texts), self.batch_size):
            lote = texts[i:i + self.batch_size]

            inputs = self.tokenizer(
                lote,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            # .float(): la media se calcula en float32 (más estable que en fp16)
            hidden = self.model(**inputs).last_hidden_state.float()
            mask = inputs["attention_mask"].unsqueeze(-1).float()

            emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1)

            if self.normalizar:
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)

            # Cada lote sale de la GPU en cuanto se calcula, así no se acumulan
            resultados.append(emb.cpu())
            pbar.update(len(lote))

        return torch.cat(resultados).numpy()

    def encode_posts(self, texts):
        """
        Codifica una lista completa de textos ordenándolos por longitud para optimizar el procesamiento por lotes.

        Precondiciones:
            - texts es una lista de strings.

        Postcondiciones:
            - Retorna una matriz de numpy con los embeddings de los posts devueltos a su orden original.
        """
        orden = np.argsort([-len(t) for t in texts], kind="stable")
        with tqdm(total=len(texts), unit="post") as pbar:
            emb = self._encode_chunk([texts[i] for i in orden], pbar)
        resultado = np.empty_like(emb)
        resultado[orden] = emb  # devuelve cada vector a su posición original
        return resultado

    #endregion

    # region Agregación por usuario

    @staticmethod
    def aggregate_users(subject_ids, post_embeddings):
        """
        Agrega los embeddings de los posts individuales por usuario calculando su media aritmética.

        Precondiciones:
            - subject_ids e iterable/serie con los identificadores de usuario correspondientes a cada post.
            - post_embeddings es una matriz numérica con los embeddings de los posts.

        Postcondiciones:
            - Retorna un DataFrame con los usuarios y su conteo de posts (user_df), junto con la matriz de embeddings agregados por usuario.
        """

        codigos, usuarios = pd.factorize(subject_ids)  # mantiene el orden de aparición

        sumas = torch.zeros(len(usuarios), post_embeddings.shape[1])
        sumas.index_add_(
            0,
            torch.from_numpy(codigos.astype(np.int64)),
            torch.from_numpy(post_embeddings),
        )

        n_posts = np.bincount(codigos)
        medias = sumas.numpy() / n_posts[:, None].astype(np.float32)

        user_df = pd.DataFrame({"subject_id": usuarios, "n_posts": n_posts})
        return user_df, medias

    #endregion

    def fit_transform(self):
        """
        Ejecuta el pipeline completo de carga de datos, vectorización de posts y agregación por usuario.

        Precondiciones:
            - El modelo y el tokenizador se encuentran inicializados correctamente.

        Postcondiciones:
            - Asigna los embeddings resultantes a los atributos de instancia y retorna los DataFrames de posts y usuarios correspondientes.
        """

        df = self.load_data()

        # Los textos repetidos se vectorizan una sola vez
        codigos, unicos = pd.factorize(df["text"])
        print(f"Posts: {len(df)} | textos distintos: {len(unicos)}")

        emb_unicos = self.encode_posts(list(unicos))
        self.post_embeddings = emb_unicos[codigos]

        user_df, self.user_embeddings = self.aggregate_users(
            df["subject_id"], self.post_embeddings
        )

        return df, user_df
