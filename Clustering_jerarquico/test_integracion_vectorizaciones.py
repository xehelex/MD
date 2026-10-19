"""El CLI actualizado y sus experimentos deben compartir la entrada seleccionada."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse

from .cli import main


class TestIntegracionVectorizaciones(unittest.TestCase):
    def test_cli_y_experimentos_con_matrices_densas_y_dispersas(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporal:
            raiz = Path(temporal)
            X = np.random.default_rng(31).random((8, 3))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            for formato in ("densa", "dispersa"):
                with self.subTest(formato=formato):
                    entrada, salida = raiz / formato, raiz / (formato + "_salida")
                    entrada.mkdir()
                    if formato == "densa":
                        np.save(entrada / "X.npy", X)
                    else:
                        sparse.save_npz(entrada / "X.npz", sparse.csr_matrix(X))
                    config = {
                        "entrada": "entrada_inexistente", "salida": "salida_no_utilizada",
                        "distancia": "cosine", "enlaces": ["average", "ward"],
                        "k_min": 2, "k_max": 3, "generar_graficas": False,
                        "incluir_db": True, "ejecutar_experimentos": True,
                        "experimentos": {"criterios": ["silhouette", "davies_bouldin", "calinski_harabasz"]},
                    }
                    with patch("Clustering_jerarquico.cli.cargar_configuracion", return_value=config), \
                            patch("Clustering_jerarquico.experimentos.cargar_configuracion", return_value=config), \
                            patch("Clustering_jerarquico.experimentos.guardar_busqueda_k"), redirect_stdout(io.StringIO()):
                        estado = main(["--entrada", str(entrada), "--salida", str(salida)])
                    self.assertEqual(estado, 0)
                    resumen = json.loads((salida / "resumen_metricas.json").read_text(encoding="utf-8"))
                    self.assertEqual(resumen["instancias"], 8)
                    self.assertEqual(resumen["atributos"], 3)
                    self.assertEqual(resumen["entrada"], str(entrada))
                    self.assertEqual([r["distancia_efectiva"] for r in resumen["resultados"]], ["cosine", "euclidean"])
                    self.assertTrue(all(np.isfinite(r["calinski_harabasz_k_final"]) for r in resumen["resultados"]))
                    archivo = next((salida / "experimentos").rglob("resultados.json"))
                    experimentos = json.loads(archivo.read_text(encoding="utf-8"))
                    self.assertEqual(experimentos["parametros"]["entrada"], str(entrada))
                    self.assertEqual(experimentos["parametros"]["instancias"], 8)
                    self.assertEqual(len(experimentos["resumen"]), 6)


if __name__ == "__main__":
    unittest.main()
