"""Pruebas de la elección de k: python -m unittest experimentos.test_eleccion_k."""

import unittest

import numpy as np

from Clustering_jerarquico.algoritmo import ClusteringJerarquico
from experimentos.eleccion_k import (
    buscar_k_biseccion, buscar_k_por_bloques, inconsistencia, k_max_recomendado,
    salto_mayor, sugerir_k,
)


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
        """Las tres formas de pedir k rechazan lo mismo: k<2, rango al reves y k>=n."""
        resultado = ClusteringJerarquico(enlace="ward", distancia="euclidean").fit(grupos_separados(3))
        casos = [
            (sugerir_k, (resultado,), {"k_min": 1}),
            (sugerir_k, (resultado,), {"k_min": 8, "k_max": 4}),
            (sugerir_k, (resultado,), {"k_max": 10_000}),
            (buscar_k_biseccion, (lambda k: 0.0,), {"k_min": 1, "k_max": 10}),
            (buscar_k_biseccion, (lambda k: 0.0,), {"k_min": 9, "k_max": 9}),
            (buscar_k_por_bloques, (lambda k: 0.0,), {"k_min": 1, "k_max": 10}),
            (buscar_k_por_bloques, (lambda k: 0.0,), {"k_min": 2, "k_max": 10, "bloques": 1}),
        ]
        for funcion, args, kwargs in casos:
            with self.subTest(funcion.__name__, **kwargs):
                with self.assertRaises(ValueError):
                    funcion(*args, **kwargs)


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




class TestBiseccion(unittest.TestCase):
    def test_encuentra_el_maximo_de_una_curva_con_un_pico(self):
        # Curva clara: sube hasta k=7 y baja. La biseccion deberia clavarlo.
        k_bueno = 7
        mejor, _, evaluaciones = buscar_k_biseccion(lambda k: -abs(k - k_bueno), 2, 15)
        self.assertEqual(mejor, k_bueno)
        self.assertLess(evaluaciones, 14)

    def test_gasta_menos_evaluaciones_que_el_barrido(self):
        llamadas = []
        buscar_k_biseccion(lambda k: (llamadas.append(k), -abs(k - 9))[1], 2, 60)
        self.assertLess(len(set(llamadas)), 15)   # el barrido serian 59

    def test_k_max_por_la_regla_de_la_raiz(self):
        self.assertEqual(k_max_recomendado(480), 15)
        self.assertEqual(k_max_recomendado(200), 10)
        self.assertGreaterEqual(k_max_recomendado(3), 2)

class TestPorBloques(unittest.TestCase):
    @staticmethod
    def dos_picos(k):
        """Pico pequeno en k=5 y el bueno en k=45. Pensada para enganar a la biseccion."""
        return max(1.40 - abs(k - 5) * 0.10, 1.50 - abs(k - 45) * 0.03, 0.1)

    def test_encuentra_el_pico_bueno_donde_la_biseccion_se_pierde(self):
        por_bloques, _, evaluaciones = buscar_k_por_bloques(self.dos_picos, 2, 60, bloques=6)
        biseccion, _, _ = buscar_k_biseccion(self.dos_picos, 2, 60)

        self.assertEqual(por_bloques, 45)
        self.assertNotEqual(biseccion, 45)   # se queda en el pico de la izquierda
        self.assertLess(evaluaciones, 40)    # 33 de 59: acierta y aun ahorra un tercio


if __name__ == "__main__":
    unittest.main()
