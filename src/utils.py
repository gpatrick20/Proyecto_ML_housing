from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_config():
    """Lee la configuración del proyecto."""
    ruta_config = ROOT / "config" / "config.yaml"

    with ruta_config.open("r", encoding="utf-8") as archivo:
        return yaml.safe_load(archivo)


def load_raw(**kwargs):
    """Carga el dataset original sin aplicar limpieza."""
    config = load_config()
    ruta_csv = ROOT / config["data"]["raw_path"]

    if not ruta_csv.is_file():
        raise FileNotFoundError(
            f"No se encontró el dataset en: {ruta_csv}"
        )

    return pd.read_csv(ruta_csv, **kwargs)