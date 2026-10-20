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

    Entrada (CSV): subject_id, title, text   (el resto de columnas se ignoran)

    Tras fit_transform() quedan disponibles:
        self.post_embeddings -> matriz (posts, dim), mismo orden que post_df
        self.user_embeddings -> matriz (usuarios, dim), mismo orden que user_df
    """

    def __init__(self,
        data_path="output/data.csv",
        model_name="FacebookAI/roberta-large",
        batch_size=32,
        max_length=512,
        preprocess=True
    ):
        self.data_path = Path(data_path)
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.preprocess = preprocess

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
        Limpieza ligera. NO quita stopwords, negaciones ni lematiza:
        RoBERTa necesita el texto natural para entender el contexto.
        """
        text = text.strip()

        if self.preprocess:
            text = html.unescape(text)       # &amp; &#x200B; ...
            text = RE_URL.sub(" ", text)     # las URLs gastan muchos tokens
            text = RE_USUARIO.sub(" ", text)
            text = RE_ESPACIOS.sub(" ", text).strip()

        return text

    def load_data(self):
        """Lee el CSV y devuelve un DataFrame con subject_id y text."""

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
        """Media de los embeddings de tokens de cada texto (ignora el padding)."""
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

            # Cada lote sale de la GPU en cuanto se calcula, así no se acumulan
            resultados.append(emb.cpu())
            pbar.update(len(lote))

        return torch.cat(resultados).numpy()

    def encode_posts(self, texts):
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
        """Media de los embeddings de los posts de cada usuario."""

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
        Ejecuta el pipeline. Devuelve:
            post_df: un registro por post (subject_id, text)
            user_df: un registro por usuario (subject_id, n_posts)
        Los embeddings quedan en self.post_embeddings y self.user_embeddings.
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
