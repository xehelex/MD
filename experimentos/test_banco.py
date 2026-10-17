"""Pruebas del banco: python -m unittest experimentos.test_banco."""

import unittest

import numpy as np

from .banco import equilibrio, grupos_utiles, recorrer_rejilla


class TestEquilibrio(unittest.TestCase):
    def test_grupos_iguales_valen_uno(self):
        # Entropia maxima: todos los grupos del mismo tamano.
        self.assertAlmostEqual(equilibrio(np.array([0, 0, 1, 1, 2, 2, 3, 3])), 1.0, places=12)

    def test_la_particion_degenerada_se_acerca_a_cero(self):
        # El caso real que motivo la metrica: 477 instancias en un grupo y tres
        # sueltas. Formalmente son cuatro grupos, en la practica uno.
        etiquetas = np.array([0] * 477 + [1, 2, 3])
        self.assertLess(equilibrio(etiquetas), 0.05)

    def test_un_solo_grupo_no_tiene_equilibrio(self):
        self.assertEqual(equilibrio(np.array([0, 0, 0, 0])), 0.0)

    def test_ordena_como_se_espera(self):
        repartida = np.array([0] * 25 + [1] * 25 + [2] * 25 + [3] * 25)
        intermedia = np.array([0] * 70 + [1] * 20 + [2] * 5 + [3] * 5)
        degenerada = np.array([0] * 97 + [1, 2, 3])
        valores = [equilibrio(x) for x in (degenerada, intermedia, repartida)]
        self.assertEqual(valores, sorted(valores))


class TestGruposUtiles(unittest.TestCase):
    def test_cuenta_los_que_pasan_el_minimo(self):
        etiquetas = np.array([0] * 30 + [1] * 12 + [2] * 4 + [3])
        utiles, sueltas = grupos_utiles(etiquetas, minimo=10)
        self.assertEqual((utiles, sueltas), (2, 5))


class TestRejilla(unittest.TestCase):
    def test_una_fila_por_combinacion_y_k(self):
        X = np.random.default_rng(0).random((20, 5))
        combinaciones = [("average", "cosine", 2.0), ("ward", "euclidean", 2.0)]
        filas = recorrer_rejilla({"prueba": X}, combinaciones, ks=[2, 3, 4], minimo=5)

        self.assertEqual(len(filas), 2 * 3)
        self.assertEqual({f["enlace"] for f in filas}, {"average", "ward"})
        for fila in filas:
            with self.subTest(enlace=fila["enlace"], k=fila["k"]):
                self.assertTrue(0 < fila["mayor_pct"] <= 100)
                self.assertTrue(0 <= fila["equilibrio"] <= 1)

    def test_ignora_los_k_que_no_caben(self):
        X = np.random.default_rng(1).random((6, 3))
        filas = recorrer_rejilla({"prueba": X}, [("ward", "euclidean", 2.0)], ks=[2, 5, 6, 9], minimo=2)
        self.assertEqual([f["k"] for f in filas], [2, 5])


if __name__ == "__main__":
    unittest.main()
