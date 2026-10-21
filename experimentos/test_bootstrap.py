"""Pruebas del bootstrap: python -m unittest experimentos.test_bootstrap."""

import unittest

import numpy as np

from Clustering_jerarquico.algoritmo import ClusteringJerarquico

from experimentos.bootstrap import bootstrap, bootstrap_por_k, estabilidad_por_grupo


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

    def test_reaprovechar_el_arbol_no_cambia_el_resultado(self):
        # bootstrap_por_k corta un mismo arbol en todos los k en vez de
        # reconstruirlo para cada uno. Si no sale identico a bootstrap(), la
        # version rapida de la rejilla esta midiendo otra cosa.
        X = grupos_separados(k=4, por_grupo=25)
        arbol = ClusteringJerarquico(enlace="ward", distancia="euclidean").fit(X)
        por_k = bootstrap_por_k(X, arbol, "ward", "euclidean", 2.0, [3, 4, 6], repeticiones=10)
        for k, (media, ancho) in por_k.items():
            with self.subTest(k=k):
                suelto = bootstrap(X, "ward", "euclidean", k=k, repeticiones=10)
                self.assertAlmostEqual(media, suelto.media, places=12)
                self.assertAlmostEqual(ancho, suelto.ancho(), places=12)

    def test_un_grupo_de_verdad_reaparece_y_el_inventado_no(self):
        X = grupos_separados(k=4, por_grupo=25)
        arbol = ClusteringJerarquico(enlace="ward", distancia="euclidean").fit(X)
        # Con 4 grupos separados y k=4 los cuatro tienen que aguantar.
        for g in estabilidad_por_grupo(X, arbol, "ward", "euclidean", 4, repeticiones=10):
            with self.subTest(grupo=g.grupo):
                self.assertGreater(g.jaccard, 0.85)

        # Partiendo los mismos datos en 8, la mitad de los cortes son inventados.
        arbol8 = ClusteringJerarquico(enlace="ward", distancia="euclidean").fit(X)
        peor = min(estabilidad_por_grupo(X, arbol8, "ward", "euclidean", 8,
                                         repeticiones=10), key=lambda g: g.jaccard)
        self.assertLess(peor.jaccard, 0.85)

    def test_parametros_invalidos(self):
        X = grupos_separados()
        for kwargs in ({"k": 1}, {"k": len(X)}, {"k": 4, "repeticiones": 1}):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    bootstrap(X, "ward", "euclidean", **kwargs)


if __name__ == "__main__":
    unittest.main()
