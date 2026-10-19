"""Pruebas del clustering jerárquico: python -m unittest Clustering_jerarquico.test_algoritmo.

Los valores esperados salen de cuentas hechas a mano o de una implementación lenta
pero evidente escrita aquí mismo. Para el contraste contra scipy está validacion.py.
"""

import unittest

import numpy as np

from .algoritmo import ClusteringJerarquico
from .cli import preparar_modelo
from .metricas import adjusted_rand_index, calinski_harabasz

# Cuatro puntos sobre una recta. Las alturas de fusión salen a mano sin esfuerzo
# y cada enlace da una secuencia distinta, que es justo lo que queremos vigilar.
RECTA = np.array([[0.0], [1.0], [4.0], [9.0]])

# Triángulo rectángulo 3-4-5, para las distancias.
TRIANGULO = np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 10.0]])


def ward_por_fuerza_bruta(X: np.ndarray, k: int) -> np.ndarray:
    """Ward tal y como lo define su criterio, sin fórmula de actualización.

    En cada paso prueba todas las uniones posibles y se queda con la que menos
    aumenta la suma de cuadrados interna. Es O(n⁴) y solo vale para ejemplos de
    juguete, pero por eso mismo sirve de referencia: si el atajo de Lance-Williams
    coincide con esto, el atajo está bien.
    """
    grupos = [[i] for i in range(len(X))]

    def suma_cuadrados(indices):
        puntos = X[indices]
        return float(((puntos - puntos.mean(axis=0)) ** 2).sum())

    while len(grupos) > k:
        mejor, menor_aumento = None, np.inf
        for a in range(len(grupos)):
            for b in range(a + 1, len(grupos)):
                aumento = (suma_cuadrados(grupos[a] + grupos[b])
                           - suma_cuadrados(grupos[a]) - suma_cuadrados(grupos[b]))
                if aumento < menor_aumento - 1e-12:
                    mejor, menor_aumento = (a, b), aumento
        a, b = mejor
        grupos[a] = grupos[a] + grupos.pop(b)

    etiquetas = np.empty(len(X), dtype=int)
    for etiqueta, miembros in enumerate(grupos):
        for miembro in miembros:
            etiquetas[miembro] = etiqueta
    return etiquetas


class TestDistancias(unittest.TestCase):
    def test_triangulo_conocido(self):
        esperado = {
            "manhattan": [[0, 7, 10], [7, 0, 9], [10, 9, 0]],
            "chebyshev": [[0, 4, 10], [4, 0, 6], [10, 6, 0]],
            "euclidean": [[0, 5, 10], [5, 0, np.sqrt(45)], [10, np.sqrt(45), 0]],
        }
        for tipo, matriz in esperado.items():
            with self.subTest(distancia=tipo):
                obtenido = ClusteringJerarquico._distancias(TRIANGULO, tipo)
                np.testing.assert_allclose(obtenido, matriz, atol=1e-12)

    def test_minkowski_p2_es_la_euclidea(self):
        np.testing.assert_allclose(
            ClusteringJerarquico._distancias(TRIANGULO, "minkowski", 2.0),
            ClusteringJerarquico._distancias(TRIANGULO, "euclidean"), atol=1e-12)

    def test_coseno_no_mira_la_longitud_del_vector(self):
        X = np.array([[1.0, 0.0], [5.0, 0.0], [0.0, 3.0]])
        D = ClusteringJerarquico._distancias(X, "cosine")
        self.assertAlmostEqual(D[0, 1], 0.0, places=12)   # misma dirección
        self.assertAlmostEqual(D[0, 2], 1.0, places=12)   # perpendiculares


class TestAgrupacion(unittest.TestCase):
    def test_alturas_sobre_la_recta(self):
        esperado = {
            "single": [1.0, 3.0, 5.0],
            "complete": [1.0, 4.0, 9.0],
            "average": [1.0, 3.5, 22 / 3],
        }
        for enlace, alturas in esperado.items():
            with self.subTest(enlace=enlace):
                resultado = ClusteringJerarquico(enlace=enlace, distancia="euclidean").fit(RECTA)
                np.testing.assert_allclose(resultado.merges[:, 2], alturas, atol=1e-12)

    def test_ward_sigue_el_criterio_de_suma_de_cuadrados(self):
        azar = np.random.default_rng(11).random((12, 3))
        resultado = ClusteringJerarquico(enlace="ward", distancia="euclidean").fit(azar)
        for k in range(2, 7):
            with self.subTest(k=k):
                referencia = ward_por_fuerza_bruta(azar, k)
                self.assertGreater(adjusted_rand_index(resultado.etiquetas(k), referencia), 0.999)

    def test_el_corte_devuelve_los_grupos_pedidos(self):
        X = np.random.default_rng(1).random((15, 4))
        resultado = ClusteringJerarquico(enlace="average", distancia="euclidean").fit(X)
        for k in range(1, len(X) + 1):
            self.assertEqual(len(np.unique(resultado.etiquetas(k))), k)
        for k in (0, len(X) + 1):
            with self.assertRaises(ValueError):
                resultado.etiquetas(k)


class TestUso(unittest.TestCase):
    def test_asignar_nuevo_va_al_grupo_mas_cercano(self):
        resultado = ClusteringJerarquico(enlace="average", distancia="euclidean").fit(RECTA)
        etiquetas = resultado.etiquetas(2)
        distancias = np.abs(RECTA.ravel() - 0.5)       # un punto pegado al 0 y al 1
        self.assertEqual(resultado.asignar_nuevo(distancias, 2), etiquetas[0])

    def test_sparse_y_denso_dan_el_mismo_arbol(self):
        from scipy import sparse
        denso = np.random.default_rng(2).random((12, 6))
        denso[denso < 0.6] = 0.0
        for enlace in ("single", "complete", "average"):
            with self.subTest(enlace=enlace):
                a = ClusteringJerarquico(enlace=enlace).fit(denso)
                b = ClusteringJerarquico(enlace=enlace).fit(sparse.csr_matrix(denso))
                np.testing.assert_allclose(a.merges, b.merges, atol=1e-10)

    def test_es_determinista(self):
        # El aglomerativo no tiene inicialización aleatoria: dos ejecuciones sobre
        # los mismos datos tienen que dar el mismo árbol, bit a bit.
        X = np.random.default_rng(4).random((13, 5))
        for enlace in ("single", "complete", "average", "ward"):
            with self.subTest(enlace=enlace):
                uno = ClusteringJerarquico(enlace=enlace, distancia="euclidean").fit(X)
                otro = ClusteringJerarquico(enlace=enlace, distancia="euclidean").fit(X)
                np.testing.assert_array_equal(uno.merges, otro.merges)

    def test_combinaciones_que_no_valen(self):
        casos = [
            {"enlace": "centroide"},
            {"distancia": "jaccard"},
            {"distancia": "minkowski", "p": 0.5},
            {"enlace": "ward", "distancia": "cosine"},
            {"enlace": "ward", "distancia": "manhattan"},
            {"enlace": "ward", "distancia": "minkowski", "p": 3.0},
        ]
        for kwargs in casos:
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    ClusteringJerarquico(**kwargs)


class TestCalinskiHarabasz(unittest.TestCase):
    def test_valor_hecho_a_mano(self):
        # Centroides 0.5 y 10.5, media global 5.5: B = 100, W = 1, n = 4, k = 2.
        X = np.array([[0.0], [1.0], [10.0], [11.0]])
        self.assertAlmostEqual(calinski_harabasz(X, np.array([0, 0, 1, 1])), 200.0, places=9)

    def test_sparse_y_denso_coinciden(self):
        from scipy import sparse
        X = np.random.default_rng(5).random((20, 4))
        etiquetas = np.arange(20) % 3
        self.assertAlmostEqual(calinski_harabasz(X, etiquetas),
                               calinski_harabasz(sparse.csr_matrix(X), etiquetas), places=9)

    def test_casos_degenerados(self):
        X = np.random.default_rng(6).random((5, 2))
        self.assertTrue(np.isnan(calinski_harabasz(X, np.zeros(5, dtype=int))))
        self.assertTrue(np.isnan(calinski_harabasz(X, np.arange(5))))


class TestWardConCoseno(unittest.TestCase):
    def test_ward_con_coseno_pasa_a_euclidea_normalizada(self):
        X = np.random.default_rng(7).random((10, 3)) * 5
        modelo, X_modelo = preparar_modelo("ward", "cosine", X)
        self.assertEqual(modelo.distancia, "euclidean")
        np.testing.assert_allclose(np.linalg.norm(np.asarray(X_modelo), axis=1), 1.0, atol=1e-12)
        modelo.fit(X_modelo)  # no debe lanzar

    def test_otros_enlaces_no_se_tocan(self):
        X = np.random.default_rng(8).random((6, 2))
        modelo, X_modelo = preparar_modelo("average", "cosine", X)
        self.assertEqual(modelo.distancia, "cosine")
        self.assertIs(X_modelo, X)


if __name__ == "__main__":
    unittest.main()
