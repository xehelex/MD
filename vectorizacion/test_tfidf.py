"""Pruebas del contrato de vectorización: python -m unittest vectorizacion.test_tfidf."""

import io
import unittest

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .tfidf import crear_vectorizador


class TestTfidf(unittest.TestCase):
    def test_caracteres_exactos_sin_cruzar_publicaciones(self):
        modelo = crear_vectorizador(min_df=1, representacion="caracteres4")
        modelo.fit_transform([("gambling", "xy"), ("abcd",)])
        self.assertEqual(set(modelo.vocabulary_), {"gamb", "ambl", "mbli", "blin", "ling", "abcd"})

    def test_espacios_y_puntuacion(self):
        modelo = crear_vectorizador(min_df=1, representacion="caracteres4")
        modelo.fit_transform([("AB C!",)])
        self.assertEqual(set(modelo.vocabulary_), {"ab c", "b c!"})

    def test_min_df_cuenta_usuarios_no_publicaciones(self):
        modelo = crear_vectorizador(min_df=2, representacion="caracteres4")
        modelo.fit_transform([("abcd wxyz", "abcd wxyz"), ("abcd",)])
        self.assertEqual(set(modelo.vocabulary_), {"abcd"})

    def test_pesos_coinciden_con_sklearn(self):
        textos = ["Gambling again!", "No gambling today", "Football today"]
        modelo = crear_vectorizador(min_df=1, representacion="caracteres4")
        obtenido = modelo.fit_transform([(t,) for t in textos])
        referencia = TfidfVectorizer(analyzer="char", ngram_range=(4, 4), dtype=np.float32)
        esperado = referencia.fit_transform(textos)
        np.testing.assert_array_equal(modelo.get_feature_names_out(), referencia.get_feature_names_out())
        np.testing.assert_allclose(obtenido.toarray(), esperado.toarray(), rtol=1e-6)

    def test_transform_y_persistencia(self):
        modelo = crear_vectorizador(min_df=1, representacion="caracteres4")
        x = modelo.fit_transform([("gambling",), ("gambling today",)])
        self.assertEqual(x.format, "csr")
        np.testing.assert_allclose(np.sqrt(x.multiply(x).sum(axis=1)).A1, 1, rtol=1e-6)
        vocabulario = dict(modelo.vocabulary_)
        idf = modelo.idf_.copy()
        self.assertEqual(modelo.transform([("zzzz",)]).nnz, 0)
        self.assertEqual(modelo.vocabulary_, vocabulario)
        np.testing.assert_array_equal(modelo.idf_, idf)
        buffer = io.BytesIO()
        joblib.dump(modelo, buffer)
        buffer.seek(0)
        copia = joblib.load(buffer)
        np.testing.assert_allclose(copia.transform([("gambling",)]).toarray(), x[:1].toarray())

    def test_palabras_sigue_disponible(self):
        modelo = crear_vectorizador(min_df=1, ngramas=2)
        modelo.fit_transform([("alpha beta", "gamma delta")])
        self.assertIn("alpha beta", modelo.vocabulary_)
        self.assertNotIn("beta gamma", modelo.vocabulary_)


if __name__ == "__main__":
    unittest.main()
