"""Pruebas del bootstrap: python -m unittest experimentos.test_bootstrap."""

import unittest

import numpy as np

from experimentos.bootstrap import bootstrap


def grupos_separados(k=4, por_grupo=25, semilla=0):
    azar = np.random.default_rng(semilla)
    centros = azar.normal(0, 25, (k, 5))
    return np.vstack([c + azar.normal(0, 1, (por_grupo, 5)) for c in centros])


class TestBootstrap(unittest.TestCase):
    def test_deja_fuera_un_tercio_de_la_muestra(self):
        # Al coger n con reemplazo, la probabilidad de que uno no salga es
        # (1-1/n)^n -> 1/e. Asi que quedan ~63% de unicos. Si esto cambia, el
        # remuestreo no es con reemplazo y el bootstrap no es bootstrap.
        X = grupos_separados()
        resultado = bootstrap(X, "ward", "euclidean", k=4, repeticiones=20)
        self.assertAlmostEqual(resultado.unicos_medios / len(X), 0.632, delta=0.04)

    def test_una_estructura_clara_sale_estable(self):
        X = grupos_separados(k=4, por_grupo=30)
        resultado = bootstrap(X, "ward", "euclidean", k=4, repeticiones=20)
        self.assertGreater(resultado.media, 0.8)

    def test_el_ruido_sale_inestable(self):
        # Sin estructura, cada remuestreo inventa grupos distintos.
        X = np.random.default_rng(3).random((100, 5))
        resultado = bootstrap(X, "ward", "euclidean", k=4, repeticiones=20)
        self.assertLess(resultado.media, 0.6)

    def test_el_intervalo_contiene_la_media(self):
        X = grupos_separados()
        resultado = bootstrap(X, "ward", "euclidean", k=4, repeticiones=20)
        bajo, alto = resultado.intervalo()
        self.assertLessEqual(bajo, resultado.media)
        self.assertLessEqual(resultado.media, alto)

    def test_parametros_invalidos(self):
        X = grupos_separados()
        for kwargs in ({"k": 1}, {"k": len(X)}, {"k": 4, "repeticiones": 1}):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    bootstrap(X, "ward", "euclidean", **kwargs)


if __name__ == "__main__":
    unittest.main()
