"""Contrato completo CSV -> vectores -> entrada de experimentos."""

from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from .__main__ import main
from .tfidf import cargar_historiales_csv


class TestCompleto(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.tmp.cleanup)
        self.raiz = Path(self.tmp.name)
        self.csv = self.raiz / "datos.csv"
        pd.DataFrame([
            {"subject_id": "u3", "text_Clustering": "football score team"},
            {"subject_id": "u1", "text_Clustering": "gambling debt help"},
            {"subject_id": "u2", "text_Clustering": "gambling help recovery"},
            {"subject_id": "u4", "text_Clustering": "football team match"},
            {"subject_id": "u1", "text_Clustering": "not gambling today"},
            {"subject_id": "vacio", "text_Clustering": " "},
            {"subject_id": "", "text_Clustering": "unknown user"},
        ]).to_csv(self.csv, index=False)

    def ejecutar(self, nombre="palabras"):
        salida = self.raiz / nombre
        with redirect_stdout(io.StringIO()):
            main(["--datos", str(self.csv), "--salida", str(salida),
                  "--representacion", nombre, "--min-df", "1"])
        return salida

    def test_todos_los_usuarios_y_mismo_orden_en_ambas_representaciones(self):
        salidas = [self.ejecutar(nombre) for nombre in ("palabras", "caracteres4")]
        filas = []
        for salida in salidas:
            self.assertEqual({p.name for p in salida.iterdir()},
                             {"X.npz", "usuarios.json", "vectorizador.joblib", "resumen.json"})
            ids = json.loads((salida / "usuarios.json").read_text(encoding="utf-8"))
            filas.append(ids)
            self.assertEqual([u["usuario"] for u in ids], ["u1", "u2", "u3", "u4"])
            x = load_npz(salida / "X.npz")
            modelo = joblib.load(salida / "vectorizador.joblib")
            docs = [u.publicaciones for u in cargar_historiales_csv(self.csv)]
            np.testing.assert_allclose(x.toarray(), modelo.transform(docs).toarray())
            np.testing.assert_allclose(np.sqrt(x.multiply(x).sum(axis=1)).A1, 1, rtol=1e-6)
            resumen = json.loads((salida / "resumen.json").read_text(encoding="utf-8"))
            self.assertEqual(resumen["usuarios"], 4)
            self.assertEqual(resumen["entrada"]["usuarios_descartados_sin_texto"], ["vacio"])
            self.assertEqual(resumen["entrada"]["filas_descartadas"], 2)
            self.assertNotIn("test_size", resumen["configuracion"])
            self.assertNotIn("seed", resumen["configuracion"])
        self.assertEqual(filas[0], filas[1])

    def test_orden_no_depende_del_orden_del_csv(self):
        antes = [u.usuario for u in cargar_historiales_csv(self.csv)]
        df = pd.read_csv(self.csv, keep_default_na=False)
        df.iloc[::-1].to_csv(self.csv, index=False)
        self.assertEqual(antes, [u.usuario for u in cargar_historiales_csv(self.csv)])

    def test_idf_usa_todos_los_usuarios(self):
        salida = self.ejecutar()
        modelo = joblib.load(salida / "vectorizador.joblib")
        # 'debt' aparece en uno de los cuatro usuarios.
        self.assertAlmostEqual(float(modelo.idf_[modelo.vocabulary_["debt"]]), np.log(5 / 2) + 1, places=6)

    def test_no_sobrescribe_resultados(self):
        salida = self.ejecutar()
        contenido = (salida / "X.npz").read_bytes()
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.ejecutar()
        self.assertEqual(contenido, (salida / "X.npz").read_bytes())

    def test_rechaza_opciones_de_separacion(self):
        for opcion in ("--test-size", "--seed"):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main([opcion, "1"])

    def test_experimentos_consume_x_completa(self):
        from experimentos.__main__ import main as experimentar

        salida = self.ejecutar()
        destino = self.raiz / "experimento"
        with redirect_stdout(io.StringIO()):
            experimentar(["--entrada", str(salida), "--salida", str(destino),
                          "--k-min", "2", "--k-max", "2", "--minimo", "1"])
        resumen = json.loads((destino / "resumen.json").read_text(encoding="utf-8"))
        self.assertEqual(resumen["matrices"]["palabras"][0], 4)
        self.assertEqual(resumen["configuraciones_evaluadas"], 5)

    def test_entrada_jerarquica_carga(self):
        from Clustering_jerarquico.__main__ import main as jerarquico

        config = json.loads((Path(__file__).parent.parent / "Clustering_jerarquico" / "config.json").read_text(encoding="utf-8"))
        self.assertTrue(config["entrada"].startswith("resultados_tfidf/datos_1_a_600/"))
        self.assertTrue((Path(__file__).parent.parent / config["entrada"]).is_dir())
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as salida:
            jerarquico(["--help"])
        self.assertEqual(salida.exception.code, 0)

    def test_normalizacion_requerida_por_experimentos_jerarquicos(self):
        from Clustering_jerarquico.metricas import normalizar_filas
        from scipy.sparse import csr_matrix
        entrada = np.array([[3., 4.], [0., 0.]])
        esperado = np.array([[0.6, 0.8], [0., 0.]])
        np.testing.assert_allclose(normalizar_filas(entrada), esperado)
        np.testing.assert_allclose(normalizar_filas(csr_matrix(entrada)).toarray(), esperado)
        np.testing.assert_array_equal(entrada, [[3., 4.], [0., 0.]])


if __name__ == "__main__":
    unittest.main()
