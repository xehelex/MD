"""CLI que carga la configuración desde un JSON y ejecuta el clustering jerárquico."""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
from scipy.sparse import load_npz
from .algoritmo import ClusteringJerarquico
from .metricas import silhouette_medio, correlacion_cofenetica, davies_bouldin, adjusted_rand_index
from .plots import guardar_dendrograma, guardar_silhouette, guardar_proyeccion_mds


def cargar_configuracion() -> dict:
    """Carga y valida el archivo config.json ubicado junto al CLI."""
    ruta = Path(__file__).parent / "config.json"

    if not ruta.is_file():
        raise FileNotFoundError(
            f"No se encuentra el archivo de configuración en: {ruta.resolve()}"
        )

    with ruta.open("r", encoding="utf-8") as archivo:
        config = json.load(archivo)

    if not isinstance(config, dict):
        raise ValueError("La configuración tiene que ser un objeto JSON válido.")

    obligatorios = {
        "entrada",
        "salida",
        "distancia",
        "enlaces",
        "k_min",
        "k_max",
        "k_final"
    }

    faltantes = obligatorios - config.keys()
    if faltantes:
        raise ValueError(f"Faltan claves obligatorias en el config.json: {sorted(faltantes)}")

    if config["distancia"] not in {"cosine", "euclidean"}:
        raise ValueError("La distancia tiene que ser 'cosine' o 'euclidean'.")

    if not isinstance(config["enlaces"], list) or not config["enlaces"]:
        raise ValueError("'enlaces' tiene que ser una lista que no esté vacía.")

    if not (2 <= config["k_min"] <= config["k_final"] <= config["k_max"]):
        raise ValueError("Tiene que cumplirse que: 2 <= k_min <= k_final <= k_max.")

    return config


def main(argv=None):
    # Cargamos la configuración directamente desde el JSON
    try:
        config = cargar_configuracion()
    except (FileNotFoundError, ValueError) as e:
        print(f"Error de configuración: {e}")
        return

    # Rutas relativas a la raíz del proyecto (MD)
    raiz = Path(__file__).parent.parent
    entrada = raiz / config["entrada"]
    salida = raiz / config["salida"]

    distancia = config["distancia"]
    enlaces = config["enlaces"]
    k_min = config["k_min"]
    k_max = config["k_max"]
    k_final = config["k_final"]

    # Parámetros opcionales con valores por defecto seguros
    estabilidad_reps = config.get("repeticiones_estabilidad", 0)
    incluir_db = config.get("incluir_db", False)

    archivo = entrada / "X_train.npz"
    if not archivo.exists():
        raise FileNotFoundError(f"No existe el archivo de entrenamiento: {archivo}")

    if salida.exists() and any(salida.iterdir()):
        print("Aviso: La carpeta de salida ya existe y tiene contenido. Se sobrescribirán los archivos necesarios.")

    X = load_npz(archivo).tocsr()
    n = X.shape[0]

    if not (2 <= k_min <= k_max < n):
        raise ValueError("Se requiere que: 2 <= k_min <= k_max < número total de usuarios.")
    if not 2 <= k_final < n:
        raise ValueError("k_final tiene que estar entre 2 y n-1.")

    salida.mkdir(parents=True, exist_ok=True)
    plots = salida / "plots"
    plots.mkdir(exist_ok=True)
    resultados = []

    for enlace in enlaces:
        inicio = time.perf_counter()
        modelo = ClusteringJerarquico(enlace=enlace, distancia=distancia)
        resultado = modelo.fit(X)
        segundos = time.perf_counter() - inicio

        sil_por_k = {}
        ks = list(range(k_min, k_max + 1))
        for k in ks:
            sil_por_k[str(k)] = silhouette_medio(resultado.distancias_originales, resultado.etiquetas(k))

        k_mejor = max(ks, key=lambda k: sil_por_k[str(k)] if np.isfinite(sil_por_k[str(k)]) else -np.inf)
        labels = resultado.etiquetas(k_final)

        estabilidad = []
        if estabilidad_reps > 0:
            # En cada vuelta tiramos el 20% de las filas al azar y comparamos
            # la partición de referencia con la del reentreno usando ARI.
            rng = np.random.default_rng(42)
            n_sub = max(k_final + 1, int(np.ceil(0.8 * n)))

            for _ in range(estabilidad_reps):
                idx = np.sort(rng.choice(n, size=min(n_sub, n - 1), replace=False))
                submodelo = ClusteringJerarquico(enlace=enlace, distancia=distancia)
                subresultado = submodelo.fit(X[idx])
                etiquetas_ref = labels[idx]
                etiquetas_sub = subresultado.etiquetas(k_final)
                estabilidad.append(adjusted_rand_index(etiquetas_ref, etiquetas_sub))

        sub = plots / enlace
        sub.mkdir(exist_ok=True)
        guardar_dendrograma(resultado, sub)
        guardar_silhouette(ks, [sil_por_k[str(k)] for k in ks], sub)
        guardar_proyeccion_mds(resultado.distancias_originales, labels, sub)

        registro = {
            "enlace": enlace,
            "distancia": distancia,
            "tiempo_segundos": round(segundos, 4),
            "correlacion_cofenetica": correlacion_cofenetica(resultado),
            "silhouette_por_k": sil_por_k,
            "k_mejor_silhouette_en_rango": int(k_mejor),
            "k_final_graficado": int(k_final),
            "estabilidad_ari_por_repeticion": estabilidad,
            "estabilidad_ari_media": float(np.mean(estabilidad)) if estabilidad else None,
        }

        # Davies-Bouldin densifica la matriz; solo lo calculamos si se pide explícitamente.
        if incluir_db:
            try:
                registro["davies_bouldin_k_final"] = davies_bouldin(X, labels)
            except (MemoryError, ValueError):
                registro["davies_bouldin_k_final"] = None
        else:
            registro["davies_bouldin_k_final"] = None

        resultados.append(registro)
        np.save(salida / f"etiquetas_{enlace}_k{k_final}.npy", labels)

    resumen = {
        "instancias_train": int(n),
        "atributos": int(X.shape[1]),
        "nota": "El silhouette más alto es una buena guía numérica, pero revisa también el tamaño y sentido real de los clusters.",
        "resultados": resultados,
    }

    (salida / "resumen_metricas.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    print(f"\n¡Listo! Gráficas guardadas en: {plots.resolve()}")


if __name__ == "__main__":
    main()