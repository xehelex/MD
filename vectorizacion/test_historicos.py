"""Comprueba que alinear los históricos no recalcula ni mezcla sus vectores."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse

from .preparar_historicos import preparar
from .svd import huella


class TestHistoricos(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.tmp.cleanup)
        self.raiz = Path(self.tmp.name)
        self.entrada, self.referencia, self.salida = [self.raiz / n for n in ("viejo", "actual", "alineado")]
        self.entrada.mkdir()
        self.referencia.mkdir()
        self.train = np.array([[2, 3, 4, 0], [4, 0, 8, 1], [9, 9, 9, 9]], dtype=np.float32)
        self.test = np.array([[1, 0, 2, 0], [0, 0, 0, 0]], dtype=np.float32)
        for particion, X, ids in (("train", self.train, ["u2", "u0", "extra"]), ("test", self.test, ["u1", "u3"])):
            sparse.save_npz(self.entrada / f"X_{particion}.npz", sparse.csr_matrix(X))
            self.guardar(self.entrada / f"usuarios_{particion}.json",
                         [{"fila": i, "usuario": u, "archivo": u + ".xml"} for i, u in enumerate(ids)])
        (self.entrada / "vectorizador.joblib").write_bytes(b"modelo conservado sin volver a ajustar")
        self.guardar(self.entrada / "resumen.json", {"usuarios": 5, "configuracion": {"test_size": 0.4}})
        sparse.save_npz(self.referencia / "X.npz", sparse.eye(4, format="csr"))
        self.guardar(self.referencia / "usuarios.json", [{"fila": i, "usuario": u} for i, u in enumerate(["u3", "u0", "u2", "u1"])])

    @staticmethod
    def guardar(ruta, valor):
        ruta.write_text(json.dumps(valor), encoding="utf-8")

    def test_union_exclusion_orden_y_valores_exactos(self):
        originales = {p.name: huella(p) for p in self.entrada.iterdir()}
        resumen = preparar(self.entrada, self.referencia, self.salida)
        X = sparse.load_npz(self.salida / "X.npz")
        np.testing.assert_array_equal(X.toarray(), [self.test[1], self.train[1], self.train[0], self.test[0]])
        self.assertEqual(X.dtype, np.float32)
        self.assertEqual(resumen["usuarios_excluidos"], ["extra"])
        self.assertEqual(resumen["usuarios_vector_cero"], ["u3"])
        self.assertEqual(resumen["usuarios_ajuste_original"], 3)
        usuarios = json.loads((self.salida / "usuarios.json").read_text(encoding="utf-8"))
        self.assertEqual([u["usuario"] for u in usuarios], ["u3", "u0", "u2", "u1"])
        self.assertEqual([u["fila"] for u in usuarios], [0, 1, 2, 3])
        self.assertEqual([(u["particion_original"], u["fila_original"]) for u in usuarios], [("test", 1), ("train", 1), ("train", 0), ("test", 0)])
        self.assertEqual(originales, {p.name: huella(p) for p in self.entrada.iterdir()})
        self.assertEqual(huella(self.salida / "vectorizador.joblib"), originales["vectorizador.joblib"])
        with self.assertRaisesRegex(ValueError, "nueva o vacía"):
            preparar(self.entrada, self.referencia, self.salida)

    def test_rechaza_usuarios_ausentes(self):
        self.guardar(self.referencia / "usuarios.json", [{"fila": i, "usuario": u} for i, u in enumerate(["u3", "u0", "u2", "ausente"])])
        with self.assertRaisesRegex(ValueError, "Faltan usuarios"):
            preparar(self.entrada, self.referencia, self.salida)
        self.assertFalse(self.salida.exists())

    def test_rechaza_ids_duplicados_entre_particiones(self):
        self.guardar(self.entrada / "usuarios_test.json", [{"fila": 0, "usuario": "u2"}, {"fila": 1, "usuario": "u3"}])
        with self.assertRaisesRegex(ValueError, "duplicados"):
            preparar(self.entrada, self.referencia, self.salida)

    def test_rechaza_columnas_o_filas_incompatibles(self):
        sparse.save_npz(self.entrada / "X_test.npz", sparse.csr_matrix(np.ones((2, 3))))
        with self.assertRaisesRegex(ValueError, "columnas"):
            preparar(self.entrada, self.referencia, self.salida)
        sparse.save_npz(self.entrada / "X_test.npz", sparse.csr_matrix(self.test))
        self.guardar(self.entrada / "usuarios_test.json", [{"fila": 1, "usuario": "u1"}, {"fila": 0, "usuario": "u3"}])
        with self.assertRaisesRegex(ValueError, "Fila o ID"):
            preparar(self.entrada, self.referencia, self.salida)

    def test_comprobar_entradas_no_ejecuta_clustering(self):
        from experimentos.__main__ import main
        preparar(self.entrada, self.referencia, self.salida)
        destino = self.raiz / "experimentos"
        with patch("experimentos.__main__.recorrer_rejilla") as rejilla, redirect_stdout(io.StringIO()) as salida:
            main(["--entrada", str(self.salida), "--entrada", str(self.referencia),
                  "--comprobar-entradas", "--salida", str(destino)])
        rejilla.assert_not_called()
        self.assertIn("2 representaciones verificadas", salida.getvalue())
        self.assertFalse(destino.exists())

    def test_lanzador_incluye_ocho_entradas_y_opciones(self):
        from experimentos.todas import ENTRADAS, main
        with patch("experimentos.todas.experimentar") as ejecutar:
            main(["--comprobar-entradas", "--estabilidad", "5"])
        argumentos = ejecutar.call_args.args[0]
        entradas = [argumentos[i + 1] for i, valor in enumerate(argumentos) if valor == "--entrada"]
        self.assertEqual(entradas, list(ENTRADAS))
        self.assertEqual(len(set(entradas)), 8)
        self.assertEqual(argumentos[-3:], ["--comprobar-entradas", "--estabilidad", "5"])


if __name__ == "__main__":
    unittest.main()
