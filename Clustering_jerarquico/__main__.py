"""Punto de entrada del clustering jerárquico.

    python -m Clustering_jerarquico                pregunta qué hacer
    python -m Clustering_jerarquico experimentar   compara enlaces y busca el mejor k
    python -m Clustering_jerarquico modelo         entrena el "mejor_modelo" de config.json y guarda modelo.pkl

Los valores por defecto salen de config.json; las opciones de la línea de comandos tienen prioridad.
También se puede lanzar con el botón Run de PyCharm sobre este archivo.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    # Ejecutado como archivo suelto (Run de PyCharm): se hace importable el paquete desde su carpeta padre.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = Path(__file__).resolve().parent.name

from .algoritmo import ClusteringJerarquico, DISTANCIAS_MODELO, entrenar_modelo
from .experimentos import CRITERIOS, Progreso, clustering_por_enlaces, buscar_k_por_combinacion


def cargar_configuracion() -> dict:
    """Lee y valida el config.json que está junto a este archivo.

    Post: devuelve el diccionario de configuración. Lanza FileNotFoundError si no existe y ValueError si
          no es un JSON válido o le faltan claves, o si alguna tiene un valor no admitido.
    """
    ruta = Path(__file__).parent / "config.json"

    if not ruta.is_file():
        raise FileNotFoundError(f"No se encuentra el archivo de configuración en: {ruta.resolve()}")

    try:
        with ruta.open("r", encoding="utf-8") as archivo:
            config = json.load(archivo)
    except json.JSONDecodeError as e:
        raise ValueError(f"config.json no es un JSON válido (línea {e.lineno}, columna {e.colno}): {e.msg}")

    if not isinstance(config, dict):
        raise ValueError("La configuración tiene que ser un objeto JSON.")

    obligatorios = {"entrada", "salida", "distancia", "enlaces", "k_min", "k_max"}
    faltantes = obligatorios - config.keys()
    if faltantes:
        raise ValueError(f"Faltan claves obligatorias en config.json: {sorted(faltantes)}")

    if config["distancia"] not in DISTANCIAS_MODELO:
        raise ValueError(f"La distancia tiene que ser una de {list(DISTANCIAS_MODELO)}.")

    if not isinstance(config["enlaces"], list) or not config["enlaces"]:
        raise ValueError("'enlaces' tiene que ser una lista no vacía.")

    desconocidos = [e for e in config["enlaces"] if e not in ClusteringJerarquico.ENLACES]
    if desconocidos:
        raise ValueError(f"Enlaces no válidos: {desconocidos}. Usa: {sorted(ClusteringJerarquico.ENLACES)}")

    if not (2 <= config["k_min"] <= config["k_max"]):
        raise ValueError("Tiene que cumplirse 2 <= k_min <= k_max.")

    if not isinstance(config.get("experimentos", {}), dict):
        raise ValueError("'experimentos' tiene que ser un objeto JSON.")

    return config


def validar_mejor_modelo(config: dict, n: int) -> dict:
    """Comprueba el bloque 'mejor_modelo' de config.json (solo hace falta en el modo 'modelo').

    Pre:  n es el número de instancias de la matriz de entrada.
    Post: devuelve {'enlace', 'distancia', 'k', 'archivo'}. Lanza ValueError si falta algo o no es válido.
    """
    bloque = config.get("mejor_modelo")
    if not isinstance(bloque, dict):
        raise ValueError("falta el bloque 'mejor_modelo', con 'enlace', 'distancia' y 'k'.")

    faltantes = {"enlace", "distancia", "k"} - bloque.keys()
    if faltantes:
        raise ValueError(f"le faltan claves: {sorted(faltantes)}")
    if bloque["enlace"] not in ClusteringJerarquico.ENLACES:
        raise ValueError(f"enlace no válido. Usa: {sorted(ClusteringJerarquico.ENLACES)}")
    if bloque["distancia"] not in DISTANCIAS_MODELO:
        raise ValueError(f"la distancia tiene que ser una de {list(DISTANCIAS_MODELO)}.")
    k = bloque["k"]
    if isinstance(k, bool) or not isinstance(k, int) or not 2 <= k < n:
        raise ValueError(f"k tiene que ser un entero con 2 <= k < {n} (número de instancias).")

    return {"enlace": bloque["enlace"], "distancia": bloque["distancia"], "k": k,
            "archivo": str(bloque.get("archivo", "modelo.pkl"))}


def crear_parser(config: dict) -> argparse.ArgumentParser:
    exp = config.get("experimentos", {})
    parser = argparse.ArgumentParser(prog="python -m Clustering_jerarquico", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("modo", nargs="?", choices=["experimentar", "modelo"],
                        help="Sin modo pregunta cuál usar (si no hay más opciones).")
    parser.add_argument("--entrada", type=Path, default=Path(config["entrada"]),
                        help="Carpeta con X.npz (dispersa) o X.npy (densa).")
    parser.add_argument("--salida", type=Path, default=Path(config["salida"]),
                        help="Carpeta donde guardar resultados y modelo.pkl.")
    parser.add_argument("--depurar", action="store_true", help="Muestra el error completo si algo falla.")

    grupo = parser.add_argument_group("solo para 'experimentar'")
    grupo.add_argument("--enlaces", nargs="+", default=config["enlaces"],
                       choices=sorted(ClusteringJerarquico.ENLACES))
    grupo.add_argument("--distancias", nargs="+", default=[config["distancia"]],
                       choices=sorted(ClusteringJerarquico.DISTANCIAS),
                       help="Distancias de la búsqueda de k (el clustering por enlace usa la de config.json).")
    grupo.add_argument("--p", type=float, default=2.0, help="Exponente para la distancia minkowski.")
    grupo.add_argument("--k_min", type=int, default=config["k_min"])
    grupo.add_argument("--k_max", type=int, default=config["k_max"])
    grupo.add_argument("--criterios", nargs="+", default=exp.get("criterios", list(CRITERIOS)),
                       choices=list(CRITERIOS))
    grupo.add_argument("--busqueda", choices=["mallado", "binaria"], default=exp.get("busqueda", "mallado"))
    grupo.add_argument("--puntos_malla", type=int, default=exp.get("puntos_malla", 12),
                       help="Valores de k de la malla gruesa (solo con --busqueda mallado).")
    grupo.add_argument("--comprobar", action="store_true", default=bool(exp.get("comprobar", False)),
                       help="Evalúa todos los k del rango y compara con el resultado de la búsqueda.")
    return parser


def preguntar(texto: str, por_defecto: str = "") -> str:
    """Pregunta por consola. Si no hay entrada interactiva, devuelve `por_defecto`."""
    try:
        return input(texto).strip()
    except EOFError:
        print()
        return por_defecto


def elegir_modo() -> str:
    print("¿Qué quieres hacer?")
    print("  1) Experimentar: comparar enlaces y buscar el mejor k")
    print("  2) Generar modelo.pkl con el 'mejor_modelo' de config.json")
    while True:
        respuesta = preguntar("Elige 1 o 2 [1]: ", "1")
        if respuesta in ("", "1"):
            return "experimentar"
        if respuesta == "2":
            return "modelo"
        print("Escribe 1 o 2.")


def cargar_entrada(carpeta: Path):
    """Carga X con la función del proyecto (vectorizacion.matrices). Se importa aquí para poder dar un error claro."""
    try:
        from vectorizacion.matrices import cargar_matriz
    except ImportError:
        raise ImportError("no se encuentra el paquete 'vectorizacion'. Ejecuta desde la carpeta raíz del "
                          "proyecto, la que contiene Clustering_jerarquico y vectorizacion.")
    return cargar_matriz(carpeta)


def modo_experimentar(X, args, config, salida: Path, progreso: Progreso) -> int:
    n = X.shape[0]
    k_max = min(args.k_max, n - 1)
    if k_max < args.k_max:
        print(f"Aviso: k_max baja a {k_max}, porque tiene que ser menor que el número de instancias ({n}).")
    if not 2 <= args.k_min <= k_max:
        print(f"Error: tiene que cumplirse 2 <= k_min <= k_max < {n}.")
        return 1

    progreso.total = len(args.enlaces) * (1 + len(args.distancias))
    estabilidad = bool(config.get("calcular_estabilidad", False))
    clustering_por_enlaces(
        X, salida, args.enlaces, config["distancia"], args.k_min, k_max,
        entrada=str(args.entrada),
        estabilidad_reps=int(config.get("repeticiones_estabilidad", 0)) if estabilidad else 0,
        semilla=int(config.get("semilla", 42)),
        generar_graficas=bool(config.get("generar_graficas", True)),
        incluir_db=bool(config.get("incluir_db", False)),
        progreso=progreso,
    )
    buscar_k_por_combinacion(
        X, salida, args.distancias, args.enlaces, args.k_min, k_max,
        criterios=args.criterios, busqueda=args.busqueda, puntos_malla=args.puntos_malla,
        comprobar=args.comprobar, p=args.p, entrada=str(args.entrada), progreso=progreso,
    )
    progreso.terminar()
    print(f"Clustering por enlace guardado en: {salida.resolve()}")
    return 0


def modo_modelo(X, config, salida: Path, progreso: Progreso) -> int:
    try:
        elegido = validar_mejor_modelo(config, X.shape[0])
    except ValueError as e:
        print(f"Error en 'mejor_modelo' de config.json: {e}")
        return 1

    ruta = salida / elegido["archivo"]
    if ruta.exists():
        respuesta = preguntar(f"{ruta.name} ya existe en {salida}. ¿Sobrescribirlo? [s/N]: ", "s")
        if respuesta.lower() not in ("s", "si", "sí"):
            print("No se ha guardado nada.")
            return 0

    progreso.total = 1
    progreso.paso(f"entrenando {elegido['enlace']} / {elegido['distancia']} / k = {elegido['k']}")
    try:
        modelo = entrenar_modelo(X, elegido["enlace"], elegido["distancia"], elegido["k"])
    except ValueError as e:
        progreso.terminar()
        print(f"No se pudo entrenar el modelo: {e}")
        return 1
    progreso.terminar()

    ruta = modelo.guardar(ruta)
    print(f"\n{modelo.resumen()}")
    print(f"\nModelo guardado en: {ruta.resolve()}")
    print("Para usarlo: ModeloEntrenado.cargar(ruta) y modelo.predecir(X_nuevo). Más detalles en el README.")
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)

    try:
        config = cargar_configuracion()
    except (FileNotFoundError, ValueError) as e:
        print(f"Error de configuración: {e}")
        return 1

    args = crear_parser(config).parse_args(argv)
    modo = args.modo or (elegir_modo() if not argv else "experimentar")

    raiz = Path(__file__).parent.parent
    entrada, salida = raiz / args.entrada, raiz / args.salida
    progreso = Progreso()

    try:
        print(f"Cargando {args.entrada}...")
        try:
            X = cargar_entrada(entrada)
        except (OSError, ValueError, ImportError) as e:
            print(f"Error de entrada: {e}")
            return 1
        print(f"{X.shape[0]} instancias y {X.shape[1]} atributos.\n")

        if modo == "experimentar":
            return modo_experimentar(X, args, config, salida, progreso)
        return modo_modelo(X, config, salida, progreso)
    except KeyboardInterrupt:
        progreso.terminar()
        print("\nInterrumpido por el usuario.")
        return 130
    except Exception as e:
        progreso.terminar()
        if args.depurar:
            raise
        tarea = f" durante '{progreso.texto}'" if progreso.texto else ""
        print(f"\nError inesperado{tarea}: {type(e).__name__}: {e}")
        print("Ejecuta con --depurar para ver el detalle.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
