"""Predicciones con el Pipeline guardado (Parte 2).

El modelo guardado incluye el preprocesamiento: recibe datos crudos, con las columnas del CSV original.
"""
import sys
from pathlib import Path

import joblib
import pandas as pd

from src.train import preparar_entrada
from src.utils import MODELS

RUTA_MODELO = MODELS / "best_model.joblib"
COLUMNAS_NO_PREDICTORAS = ["SalePrice", "Order", "PID"]


def cargar_modelo(ruta=RUTA_MODELO):
    """Carga el Pipeline guardado en models/best_model.joblib."""
    ruta = Path(ruta)
    if not ruta.is_file():
        raise FileNotFoundError(
            f"No se encontró el modelo: {ruta}. Ejecuta el notebook 02 para generarlo."
        )
    return joblib.load(ruta)


def leer_datos_crudos(ruta_csv):
    """Lee un CSV con el mismo formato del original (preserva los textos 'NA' y 'None')."""
    return pd.read_csv(ruta_csv, keep_default_na=False, na_values=[""])


def predecir(datos_crudos, modelo=None):
    """Predice SalePrice (USD) a partir de datos crudos.

    datos_crudos: DataFrame con las columnas del CSV original, leído con leer_datos_crudos().
    Se ignoran SalePrice, Order y PID si están presentes; Sale Type y Sale Condition se descartan.
    """
    modelo = cargar_modelo() if modelo is None else modelo

    X = datos_crudos.drop(
        columns=[c for c in COLUMNAS_NO_PREDICTORAS if c in datos_crudos.columns]
    )
    X = preparar_entrada(X)

    esperadas = list(modelo.feature_names_in_)
    faltantes = [c for c in esperadas if c not in X.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas necesarias para predecir: {faltantes}")

    predicciones = modelo.predict(X[esperadas])
    return pd.Series(predicciones, index=datos_crudos.index, name="SalePrice_predicho")


def predecir_csv(ruta_csv, ruta_salida=None):
    """Lee un CSV crudo, predice y opcionalmente guarda el resultado."""
    resultado = predecir(leer_datos_crudos(ruta_csv))
    if ruta_salida is not None:
        resultado.to_csv(ruta_salida, header=True)
    return resultado


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Uso: python -m src.predict ruta_datos_crudos.csv [ruta_salida.csv]")
    salida = sys.argv[2] if len(sys.argv) > 2 else None
    print(predecir_csv(sys.argv[1], salida).head())
