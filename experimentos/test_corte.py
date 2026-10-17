"""Pruebas del corte por tamaño: python -m unittest experimentos.test_corte."""

import unittest

import numpy as np

from Clustering_jerarquico.algoritmo import ClusteringJerarquico
from experimentos.corte import cortar_por_tamano


def arbol_con_cola(n_nucleo=40, n_sueltas=6, semilla=0):
    """Un nucleo compacto mas unas pocas instancias lejanas y separadas entre si.

    Reproduce en pequeno lo que pasa con los datos reales: el encadenamiento
    despega esas instancias de una en una y se come el presupuesto de k.
    """
    azar = np.random.default_rng(semilla)
    nucleo = azar.normal(0, 0.4, (n_nucleo, 2))
    sueltas = np.array([[40.0 * (i + 1), 40.0 * (i + 1)] for i in range(n_sueltas)])
    return np.vstack([nucleo, sueltas])


class TestCortePorTamano(unittest.TestCase):
    def test_todas_las_instancias_reciben_etiqueta(self):
        X = arbol_con_cola()
        resultado = ClusteringJerarquico(enlace="ward", distancia="euclidean").fit(X)
        etiquetas, informe = cortar_por_tamano(resultado, minimo=5, max_grupos=4)

        self.assertEqual(len(etiquetas), len(X))
        self.assertEqual(sum(informe["tamanos"]) + informe["atipicas"], len(X))
        self.assertEqual(set(np.unique(etiquetas[etiquetas >= 0])), set(range(informe["grupos"])))

    def test_ningun_grupo_baja_del_minimo(self):
        X = arbol_con_cola()
        resultado = ClusteringJerarquico(enlace="single", distancia="euclidean").fit(X)
        for minimo in (3, 5, 10):
            with self.subTest(minimo=minimo):
                _, informe = cortar_por_tamano(resultado, minimo=minimo, max_grupos=6)
                self.assertTrue(all(t >= minimo for t in informe["tamanos"]))

    def test_no_gasta_grupos_en_instancias_sueltas(self):
        X = arbol_con_cola(n_nucleo=40, n_sueltas=6)
        resultado = ClusteringJerarquico(enlace="single", distancia="euclidean").fit(X)

        # El corte clasico dedica grupos enteros a una sola instancia.
        clasico = sorted(np.bincount(resultado.etiquetas(7)), reverse=True)
        self.assertEqual(clasico[1], 1)

        # El corte por tamano las aparta y reserva los grupos para el nucleo.
        etiquetas, informe = cortar_por_tamano(resultado, minimo=5, max_grupos=7)
        self.assertTrue(all(t >= 5 for t in informe["tamanos"]))
        self.assertTrue(all(etiquetas[i] == -1 for i in range(40, 46)))

        # Con single tambien se aparta parte del nucleo, porque el encadenamiento
        # lo va pelando: el metodo no lo arregla, lo deja a la vista.
        self.assertGreater(informe["atipicas"], 6)

    def test_minimo_uno_equivale_al_corte_clasico(self):
        X = arbol_con_cola()
        resultado = ClusteringJerarquico(enlace="ward", distancia="euclidean").fit(X)
        etiquetas, informe = cortar_por_tamano(resultado, minimo=1, max_grupos=5)

        self.assertEqual(informe["atipicas"], 0)
        self.assertEqual(sorted(informe["tamanos"], reverse=True),
                         sorted(np.bincount(resultado.etiquetas(5)), reverse=True))

    def test_parametros_invalidos(self):
        resultado = ClusteringJerarquico(enlace="ward", distancia="euclidean").fit(arbol_con_cola())
        for kwargs in ({"minimo": 0}, {"minimo": 5, "max_grupos": 0}):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    cortar_por_tamano(resultado, **kwargs)


if __name__ == "__main__":
    unittest.main()
