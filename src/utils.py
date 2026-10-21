import pandas as pd

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw" / "Ames_Iowa_Housing_Dataset.csv"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"


def load_raw(**kwargs):
    """Carga el dataset original."""
    return pd.read_csv(DATA_RAW, **kwargs)


def load_data(nombre_archivo=None):
    """
    Carga datos procesados.

    Sin argumentos:
        Devuelve X_train, X_test, y_train, y_test.

    Con nombre_archivo:
        Devuelve el DataFrame del archivo indicado.
    """
    if nombre_archivo is not None:
        ruta = DATA_PROCESSED / nombre_archivo

        if not ruta.is_file():
            raise FileNotFoundError(
                f"No se encontró el archivo: {ruta}"
            )

        return pd.read_csv(ruta)

    train = load_data("train.csv")
    test = load_data("test.csv")

    for nombre, datos in [
        ("train.csv", train),
        ("test.csv", test)
    ]:
        if "target" not in datos.columns:
            raise ValueError(
                f"El archivo {nombre} no contiene la columna 'target'."
            )

    X_train = train.drop(columns="target").copy()
    X_test = test.drop(columns="target").copy()

    y_train = train["target"].copy()
    y_test = test["target"].copy()

    if not X_train.columns.equals(X_test.columns):
        raise ValueError(
            "Los predictores de entrenamiento y prueba "
            "no tienen las mismas columnas en el mismo orden."
        )

    return X_train, X_test, y_train, y_test