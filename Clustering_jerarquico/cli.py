"""CLI que carga la configuración desde un JSON y ejecuta el clustering jerárquico."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from vectorizacion.matrices import cargar_matriz
from .algoritmo import ClusteringJerarquico
from .metricas import (silhouette_medio, correlacion_cofenetica, davies_bouldin,
                       calinski_harabasz, adjusted_rand_index, normalizar_filas)
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

    obligatorios = {"entrada", "salida", "distancia", "enlaces", "k_min", "k_max"}

    faltantes = obligatorios - config.keys()
    if faltantes:
        raise ValueError(f"Faltan claves obligatorias en el config.json: {sorted(faltantes)}")

    if config["distancia"] not in {"cosine", "euclidean"}:
        raise ValueError("La distancia tiene que ser 'cosine' o 'euclidean'.")

    if not isinstance(config["enlaces"], list) or not config["enlaces"]:
        raise ValueError("'enlaces' tiene que ser una lista que no esté vacía.")

    desconocidos = [e for e in config["enlaces"] if e not in ClusteringJerarquico.ENLACES]
    if desconocidos:
        raise ValueError(
            f"Enlaces no válidos: {desconocidos}. Usa: {sorted(ClusteringJerarquico.ENLACES)}"
        )

    if not (2 <= config["k_min"] <= config["k_max"]):
        raise ValueError("Tiene que cumplirse que: 2 <= k_min <= k_max.")

    if not isinstance(config.get("experimentos", {}), dict):
        raise ValueError("'experimentos' tiene que ser un objeto JSON.")

    return config


def preparar_modelo(enlace: str, distancia: str, X, p: float = 2.0):
    """Crea el modelo y devuelve (modelo, X_para_el_modelo).

    Ward solo admite distancia euclídea. Si se pide coseno, se normalizan las filas
    a norma 1 y se usa la euclídea: sobre vectores unitarios d_eucl² = 2·d_coseno,
    así que Ward sigue trabajando en el espacio coseno. Para el resto de combinaciones
    no se toca nada (y las no válidas lanzan ValueError desde el constructor).
    """
    if enlace == "ward" and distancia == "cosine":
        return ClusteringJerarquico(enlace=enlace, distancia="euclidean"), normalizar_filas(X)
    return ClusteringJerarquico(enlace=enlace, distancia=distancia, p=p), X


def main(argv=None) -> int:
    try:
        config = cargar_configuracion()
    except (FileNotFoundError, ValueError) as e:
        print(f"Error de configuración: {e}")
        return 1

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=Path(config["entrada"]),
                        help="Carpeta con X.npz (dispersa) o X.npy (densa).")
    parser.add_argument("--salida", type=Path, default=Path(config["salida"]),
                        help="Carpeta donde guardar el clustering y sus experimentos.")
    args = parser.parse_args(argv)
    raiz = Path(__file__).parent.parent
    entrada = raiz / args.entrada
    salida = raiz / args.salida

    distancia = config["distancia"]
    enlaces = config["enlaces"]
    k_min = config["k_min"]
    k_max = config["k_max"]

    calcular_estabilidad = bool(config.get("calcular_estabilidad", False))
    estabilidad_reps = int(config.get("repeticiones_estabilidad", 0)) if calcular_estabilidad else 0
    semilla = int(config.get("semilla", 42))
    generar_graficas = bool(config.get("generar_graficas", True))
    incluir_db = bool(config.get("incluir_db", False))
    ejecutar_experimentos = bool(config.get("ejecutar_experimentos", False))

    try:
        X = cargar_matriz(entrada)
    except (OSError, ValueError) as e:
        print(f"Error de entrada: {e}")
        return 1

    if salida.exists() and any(salida.iterdir()):
        print("Aviso: La carpeta de salida ya existe y tiene contenido. Se sobrescribirán los archivos necesarios.")

    n = X.shape[0]

    if not (2 <= k_min <= k_max < n):
        print("Se requiere que: 2 <= k_min <= k_max < número total de usuarios.")
        return 1

    salida.mkdir(parents=True, exist_ok=True)
    plots = salida / "plots"
    plots.mkdir(exist_ok=True)
    resultados = []

    for enlace in enlaces:
        modelo, X_modelo = preparar_modelo(enlace, distancia, X)
        if modelo.distancia != distancia:
            print(f"[{enlace}] {distancia} no es válida para este enlace: "
                  f"se usa {modelo.distancia} sobre filas normalizadas.")

        inicio = time.perf_counter()
        resultado = modelo.fit(X_modelo)
        segundos = time.perf_counter() - inicio

        sil_por_k = {}
        ks = list(range(k_min, k_max + 1))
        for k in ks:
            sil_por_k[str(k)] = silhouette_medio(resultado.distancias_originales, resultado.etiquetas(k))

        k_mejor = max(ks, key=lambda k: sil_por_k[str(k)] if np.isfinite(sil_por_k[str(k)]) else -np.inf)
        labels = resultado.etiquetas(k_mejor)

        estabilidad = []
        if estabilidad_reps > 0:
            rng = np.random.default_rng(semilla)
            n_sub = max(k_mejor + 1, int(np.ceil(0.8 * n)))

            for _ in range(estabilidad_reps):
                idx = np.sort(rng.choice(n, size=min(n_sub, n - 1), replace=False))
                submodelo = ClusteringJerarquico(enlace=modelo.enlace, distancia=modelo.distancia, p=modelo.p)
                subresultado = submodelo.fit(X_modelo[idx])
                estabilidad.append(adjusted_rand_index(labels[idx], subresultado.etiquetas(k_mejor)))

        if generar_graficas:
            sub = plots / enlace
            sub.mkdir(exist_ok=True)
            guardar_dendrograma(resultado, sub)
            guardar_silhouette(ks, [sil_por_k[str(k)] for k in ks], sub)
            guardar_proyeccion_mds(resultado.distancias_originales, labels, sub)

        # Las métricas basadas en centroides (DB y CH) son euclídeas: con coseno se evalúan
        # sobre filas normalizadas, igual que en experimentos.py.
        X_metricas = normalizar_filas(X_modelo) if modelo.distancia == "cosine" else X_modelo

        registro = {
            "enlace": enlace,
            "distancia": distancia,
            "distancia_efectiva": modelo.distancia,
            "tiempo_segundos": round(segundos, 4),
            "correlacion_cofenetica": correlacion_cofenetica(resultado),
            "silhouette_por_k": sil_por_k,
            "k_mejor_silhouette_en_rango": int(k_mejor),
            "estabilidad_ari_por_repeticion": estabilidad,
            "estabilidad_ari_media": float(np.mean(estabilidad)) if estabilidad else None,
            "calinski_harabasz_k_final": calinski_harabasz(X_metricas, labels),
            "davies_bouldin_k_final": None,
        }

        if incluir_db:
            try:
                registro["davies_bouldin_k_final"] = davies_bouldin(X_metricas, labels)
            except (MemoryError, ValueError) as e:
                print(f"Error calculando Davies-Bouldin para {enlace}: {e}")

        resultados.append(registro)
        np.save(salida / f"etiquetas_{enlace}_k{k_mejor}.npy", labels)

    resumen = {
        "entrada": str(args.entrada),
        "instancias": int(n),
        "atributos": int(X.shape[1]),
        "nota": "El silhouette más alto guía de forma automática el número de clusters óptimo para cada enlace.",
        "resultados": resultados,
    }

    (salida / "resumen_metricas.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    if generar_graficas:
        print(f"\n¡Listo! Gráficas guardadas en: {plots.resolve()}")

    if ejecutar_experimentos:
        # Import perezoso: experimentos importa de este módulo y un import al inicio sería circular.
        from .experimentos import main as experimentos_main
        print("\n=== Experimentos: búsqueda del mejor k ===")
        experimentos_main(["--entrada", str(entrada), "--salida", str(salida)])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
