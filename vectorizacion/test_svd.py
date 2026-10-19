"""SVD reproducible, trazabilidad de filas y consumo real por los experimentos."""

from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

from .matrices import cargar_matriz
from .svd import main, reducir, huella


class TestSVD(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.tmp.cleanup)
        self.raiz = Path(self.tmp.name)
        self.entrada = self.raiz / "tfidf"
        self.entrada.mkdir()
        self.X = normalize(sparse.csr_matrix(np.array([
            [5, 4, 1, 0, 0, 0, 1, 0], [4, 5, 1, 0, 0, 0, 0, 1],
            [0, 0, 0, 5, 4, 1, 1, 0], [0, 0, 0, 4, 5, 1, 0, 1],
            [1, 0, 2, 0, 0, 0, 5, 4], [0, 1, 1, 0, 0, 0, 4, 5],
        ], dtype=np.float32)))
        sparse.save_npz(self.entrada / "X.npz", self.X)
        self.usuarios = [{"fila": i, "usuario": f"u{i}", "publicaciones_utiles": i + 1} for i in range(6)]
        self.guardar_usuarios(self.usuarios)
        self.salida = self.raiz / "svd"

    def guardar_usuarios(self, usuarios):
        (self.entrada / "usuarios.json").write_text(json.dumps(usuarios), encoding="utf-8")

    def test_exportacion_reutilizable_y_varianza_antes_de_l2(self):
        original = huella(self.entrada / "X.npz")
        resumen = reducir(self.entrada, self.salida, 2)
        self.assertEqual({p.name for p in self.salida.iterdir()}, {"X.npy", "svd.joblib", "usuarios.json", "resumen.json"})
        X = cargar_matriz(self.salida)
        self.assertIsInstance(X, np.ndarray)
        self.assertEqual(X.shape, (6, 2))
        self.assertEqual(X.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(X, axis=1), 1, rtol=1e-6)
        modelo = joblib.load(self.salida / "svd.joblib")
        np.testing.assert_allclose(modelo.transform(self.X), X, atol=1e-6)
        # El mismo modelo sirve para una nueva fila, sin reajuste.
        nuevo = modelo.transform(self.X[:1])
        np.testing.assert_allclose(nuevo, X[:1], atol=1e-6)
        svd = modelo.named_steps["truncatedsvd"]
        esperado = np.var(svd.transform(self.X), axis=0).sum() / np.var(self.X.toarray(), axis=0).sum()
        self.assertAlmostEqual(resumen["varianza_explicada_acumulada"], float(esperado), places=6)
        self.assertEqual((self.salida / "usuarios.json").read_bytes(), (self.entrada / "usuarios.json").read_bytes())
        self.assertEqual(original, huella(self.entrada / "X.npz"))
        self.assertEqual(resumen["origen"]["archivos_sha256"]["X.npz"], original)

    def test_cli_salida_predeterminada_y_semilla_reproducible(self):
        with redirect_stdout(io.StringIO()):
            main(["--entrada", str(self.entrada), "--dimensiones", "2"])
        reducir(self.entrada, self.salida, 2)
        np.testing.assert_allclose(cargar_matriz(self.entrada.with_name("tfidf_svd2")), cargar_matriz(self.salida), atol=1e-6)

    def test_limites_y_no_sobrescritura(self):
        for dimensiones in (0, -1, 6, 100):
            with self.subTest(dimensiones=dimensiones), self.assertRaises(ValueError):
                reducir(self.entrada, self.salida, dimensiones)
            self.assertFalse(self.salida.exists())
        with self.assertRaises(ValueError):
            reducir(self.entrada, self.salida, 2, semilla=-1)
        reducir(self.entrada, self.salida, 2)
        original = huella(self.salida / "X.npy")
        with self.assertRaises(ValueError):
            reducir(self.entrada, self.salida, 2)
        self.assertEqual(original, huella(self.salida / "X.npy"))

    def test_rechaza_metadatos_desalineados(self):
        casos = [self.usuarios[:-1], list(reversed(self.usuarios)),
                 [dict(u, usuario="duplicado") for u in self.usuarios]]
        for usuarios in casos:
            with self.subTest(usuarios=usuarios):
                self.guardar_usuarios(usuarios)
                with self.assertRaises(ValueError):
                    reducir(self.entrada, self.salida, 2)
                self.assertFalse(self.salida.exists())

    def test_vector_cero_conserva_fila_y_se_informa(self):
        X = self.X.toarray()
        X[0] = 0
        sparse.save_npz(self.entrada / "X.npz", sparse.csr_matrix(X))
        resumen = reducir(self.entrada, self.salida, 2)
        self.assertEqual(resumen["usuarios_vector_cero"], ["u0"])
        np.testing.assert_array_equal(cargar_matriz(self.salida)[0], [0, 0])

    def test_cargador_rechaza_ambiguedad_y_valores_invalidos(self):
        np.save(self.entrada / "X.npy", self.X.toarray())
        with self.assertRaisesRegex(ValueError, "Hay X.npz y X.npy"):
            cargar_matriz(self.entrada)
        self.salida.mkdir()
        for X in (np.array([1, 2]), np.array([[np.nan]]), np.array([[np.inf]]), np.empty((0, 2))):
            np.save(self.salida / "X.npy", X)
            with self.assertRaises(ValueError):
                cargar_matriz(self.salida)

    def test_experimentos_compara_dispersa_y_densa_con_remuestreo(self):
        from experimentos.__main__ import main as experimentar
        reducir(self.entrada, self.salida, 2)
        destino = self.raiz / "experimento"
        with redirect_stdout(io.StringIO()):
            experimentar(["--entrada", str(self.entrada), "--entrada", str(self.salida), "--salida", str(destino),
                          "--k-min", "2", "--k-max", "2", "--minimo", "1", "--estabilidad", "2"])
        resumen = json.loads((destino / "resumen.json").read_text(encoding="utf-8"))
        self.assertEqual(resumen["matrices"], {"tfidf": [6, 8], "svd": [6, 2]})
        self.assertEqual(resumen["configuraciones_evaluadas"], 10)
        usuarios = list(reversed(self.usuarios))
        for i, u in enumerate(usuarios):
            u["fila"] = i
        (self.salida / "usuarios.json").write_text(json.dumps(usuarios), encoding="utf-8")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            experimentar(["--entrada", str(self.entrada), "--entrada", str(self.salida)])

    def test_clustering_jerarquico_acepta_densa(self):
        from Clustering_jerarquico.experimentos import main as experimentar
        from Clustering_jerarquico.cli import cargar_configuracion
        reducir(self.entrada, self.salida, 2)
        config = cargar_configuracion()
        config["salida"] = str(self.raiz / "jerarquico")
        with patch("Clustering_jerarquico.experimentos.cargar_configuracion", return_value=config), \
                patch("Clustering_jerarquico.experimentos.guardar_busqueda_k"), redirect_stdout(io.StringIO()):
            experimentar(["--entrada", str(self.salida), "--k_min", "2", "--k_max", "2", "--enlaces", "average"])
        archivo = next((self.raiz / "jerarquico").rglob("resultados.json"))
        resultado = json.loads(archivo.read_text(encoding="utf-8"))
        self.assertEqual(resultado["parametros"]["instancias"], 6)
        self.assertEqual(resultado["parametros"]["atributos"], 2)
        self.assertEqual(len(resultado["resumen"]), 2)


if __name__ == "__main__":
    unittest.main()
