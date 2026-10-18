"""Pruebas de la elección de k: python -m unittest experimentos.test_eleccion_k."""

import unittest

import numpy as np

from Clustering_jerarquico.algoritmo import ClusteringJerarquico
from experimentos.eleccion_k import inconsistencia, salto_mayor, sugerir_k


def grupos_separados(k, por_grupo=30, semilla=0):
    """k nubes bien lejos unas de otras, para saber cual es la respuesta buena."""
    azar = np.random.default_rng(semilla)
    centros = azar.normal(0, 25, (k, 6))
    return np.vstack([c + azar.normal(0, 1, (por_grupo, 6)) for c in centros])


class TestSugerirK(unittest.TestCase):
    def test_encuentra_el_k_verdadero_en_datos_sinteticos(self):
        # No clava el primero siempre, pero el bueno sale entre los dos
        # primeros. Si esto empeora, es que algo se ha torcido.
        for k_real in (3, 4, 5, 6, 8):
            with self.subTest(k=k_real):
                X = grupos_separados(k_real)
                resultado = ClusteringJerarquico(enlace="ward", distancia="euclidean").fit(X)
                propuestos = [k for k, _ in sugerir_k(resultado, k_max=20)]
                self.assertIn(k_real, propuestos[:2])

    def test_respeta_el_rango_pedido(self):
        resultado = ClusteringJerarquico(enlace="ward", distancia="euclidean").fit(grupos_separados(4))
        for k, _ in sugerir_k(resultado, k_min=5, k_max=9, cuantos=5):
            self.assertTrue(5 <= k <= 9)

    def test_rangos_imposibles(self):
        resultado = ClusteringJerarquico(enlace="ward", distancia="euclidean").fit(grupos_separados(3))
        for kwargs in ({"k_min": 1}, {"k_min": 8, "k_max": 4}, {"k_max": 10_000}):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    sugerir_k(resultado, **kwargs)


class TestSaltoMayor(unittest.TestCase):
    def test_devuelve_candidatos_ordenados_y_en_rango(self):
        X = grupos_separados(4)
        resultado = ClusteringJerarquico(enlace="ward", distancia="euclidean").fit(X)
        candidatos = salto_mayor(resultado, k_min=2, k_max=15, cuantos=5)

        puntuaciones = [s for _, s in candidatos]
        self.assertEqual(puntuaciones, sorted(puntuaciones, reverse=True))
        for k, _ in candidatos:
            self.assertTrue(2 <= k <= 15)


class TestInconsistencia(unittest.TestCase):
    def test_una_por_fusion_y_sin_valores_raros(self):
        resultado = ClusteringJerarquico(enlace="ward", distancia="euclidean").fit(grupos_separados(4))
        valores = inconsistencia(resultado)
        self.assertEqual(len(valores), len(resultado.merges))
        self.assertTrue(np.all(np.isfinite(valores)))


if __name__ == "__main__":
    unittest.main()
