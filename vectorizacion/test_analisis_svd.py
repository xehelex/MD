"""La curva debe corresponder a SVD sin centrar y localizar el primer cruce."""

from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy import sparse

from .analisis_svd import analizar, curva_varianza, localizar_umbral, main
from .svd import huella


class TestCurva(unittest.TestCase):
    def test_equivalencia_con_svd_directa_sin_centrar(self):
        rng = np.random.default_rng(42)
        # Matrices anchas y altas; la media no es cero.
        for forma in ((8, 13), (13, 8)):
            with self.subTest(forma=forma):
                A = rng.random(forma) + np.arange(forma[1])
                X = sparse.csr_matrix(A)
                antes = X.data.copy()
                por_componente, acumulada = curva_varianza(X)
                _, _, vt = np.linalg.svd(A, full_matrices=False)
                esperado = np.var(A @ vt.T, axis=0) / np.var(A, axis=0).sum()
                np.testing.assert_allclose(por_componente, esperado[:-1], rtol=1e-9, atol=1e-10)
                np.testing.assert_allclose(acumulada, np.cumsum(esperado[:-1]), rtol=1e-9, atol=1e-10)
                np.testing.assert_array_equal(X.data, antes)
                self.assertTrue(np.all(np.diff(acumulada) >= 0))
                # Confirma que no se ha sustituido por PCA de la matriz centrada.
                _, sc, _ = np.linalg.svd(A - A.mean(axis=0), full_matrices=False)
                pca = sc**2 / np.sum(sc**2)
                self.assertFalse(np.allclose(por_componente, pca[:-1]))

    def test_umbral_primer_cruce_exacto_no_alcanzado_y_meseta(self):
        curva = np.array([0.2, 0.5, 0.9, 0.9, 0.96])
        self.assertEqual(localizar_umbral(curva, 0.2), 1)
        self.assertEqual(localizar_umbral(curva, 0.90), 3)
        self.assertEqual(localizar_umbral(curva, 0.95), 5)
        self.assertIsNone(localizar_umbral(curva, 0.99))

    def test_rechaza_sin_varianza_y_entradas_invalidas(self):
        for X in (sparse.csr_matrix(np.ones((4, 6))), sparse.csr_matrix((4, 6)),
                  sparse.csr_matrix([[1, np.nan], [0, 1]]), sparse.csr_matrix([[1, 2]]), np.eye(3)):
            with self.subTest(X=X), self.assertRaises(ValueError):
                curva_varianza(X)


class TestInforme(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.tmp.cleanup)
        self.raiz = Path(self.tmp.name)
        self.entradas = []
        rng = np.random.default_rng(8)
        for nombre in ("palabras", "caracteres"):
            entrada = self.raiz / nombre
            entrada.mkdir()
            sparse.save_npz(entrada / "X.npz", sparse.csr_matrix(rng.random((8, 15))))
            usuarios = [{"fila": i, "usuario": f"u{i}"} for i in range(8)]
            (entrada / "usuarios.json").write_text(json.dumps(usuarios), encoding="utf-8")
            self.entradas.append(entrada)
        self.salida = self.raiz / "informe"

    def test_cli_exporta_curvas_tablas_y_figuras_sin_cambiar_entradas(self):
        antes = [huella(p / "X.npz") for p in self.entradas]
        with redirect_stdout(io.StringIO()):
            main(["--entrada", str(self.entradas[0]), "--entrada", str(self.entradas[1]),
                  "--salida", str(self.salida), "--dimensiones", "1", "3", "100"])
        self.assertEqual({p.name for p in self.salida.iterdir()}, {
            "curva_completa.csv", "comparativa.csv", "umbrales.csv", "resumen.json",
            "INFORME.md", "varianza_acumulada.png", "varianza_acumulada.svg"})
        informe = json.loads((self.salida / "resumen.json").read_text(encoding="utf-8"))
        self.assertEqual(len(informe["resultados"]), 2)
        for resultado in informe["resultados"]:
            self.assertEqual(resultado["dimensiones_maximas"], 7)
            self.assertEqual(resultado["dimensiones_solicitadas_fuera_de_rango"], [100])
            for punto in resultado["umbrales"]:
                k = punto["dimensiones"]
                if k is not None:
                    self.assertLess(punto["varianza_dimension_anterior"], punto["umbral"])
                    self.assertGreaterEqual(punto["varianza_acumulada"], punto["umbral"])
        with (self.salida / "curva_completa.csv").open(encoding="utf-8", newline="") as f:
            self.assertEqual(len(list(csv.DictReader(f))), 14)
        self.assertEqual(antes, [huella(p / "X.npz") for p in self.entradas])
        self.assertTrue((self.salida / "varianza_acumulada.png").read_bytes().startswith(b"\x89PNG"))
        self.assertIn("<svg", (self.salida / "varianza_acumulada.svg").read_text(encoding="utf-8"))

    def test_no_sobrescribe_ni_acepta_opciones_invalidas(self):
        self.salida.mkdir()
        marca = self.salida / "conservar.txt"
        marca.write_text("original", encoding="utf-8")
        with self.assertRaises(ValueError):
            analizar(self.entradas, self.salida)
        self.assertEqual(marca.read_text(encoding="utf-8"), "original")
        for opciones in ({"dimensiones": [0]}, {"umbrales": [1.1]}, {"umbrales": [float("nan")]}, {"umbrales": []}):
            with self.subTest(opciones=opciones), self.assertRaises(ValueError):
                analizar(self.entradas, self.raiz / "otra", **opciones)
        self.assertFalse((self.raiz / "otra").exists())

    def test_rechaza_usuarios_distintos(self):
        archivo = self.entradas[1] / "usuarios.json"
        usuarios = json.loads(archivo.read_text(encoding="utf-8"))
        usuarios[0]["usuario"] = "otro"
        archivo.write_text(json.dumps(usuarios), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "mismos usuarios"):
            analizar(self.entradas, self.salida)
        self.assertFalse(self.salida.exists())


if __name__ == "__main__":
    unittest.main()
