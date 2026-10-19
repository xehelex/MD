"""Entrada del paquete y lectura de su configuración compartida."""

import json
from pathlib import Path


def cargar_configuracion():
    """Lee config.json y comprueba los campos usados por el ejecutable."""
    ruta = Path(__file__).with_name("config.json")
    config = json.loads(ruta.read_text(encoding="utf-8"))
    requeridas = {"entrada", "salida", "distancia", "enlaces", "k_min", "k_max"}
    if not isinstance(config, dict) or not requeridas.issubset(config):
        raise ValueError("config.json debe incluir entrada, salida, distancia, enlaces, k_min y k_max.")
    return config


def main(argv=None):
    """Reutiliza el ejecutable existente de experimentos jerárquicos."""
    from .experimentos import main as ejecutar

    return ejecutar(argv)
