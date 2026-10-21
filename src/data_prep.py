from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

from src.utils import load_raw, DATA_PROCESSED, MODELS

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def cargar_y_separar():
    """Carga el dataset y separa entrenamiento y prueba."""
    df = load_raw(
        keep_default_na=False,
        na_values=[""]
    )

    target = "SalePrice"

    if target not in df.columns:
        raise ValueError(
            f"No se encontró la variable objetivo: {target}"
        )

    if df[target].isna().any():
        raise ValueError(
            "La variable objetivo contiene valores ausentes."
        )

    X = df.drop(columns=[target, "Order", "PID"]).copy()
    y = df[target].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


def limpiar_datos(X):
    """Aplica reglas explícitas sin aprender estadísticas."""
    X_limpio = X.copy()

    # Código de clase: se representa como categoría.
    X_limpio["MS SubClass"] = X_limpio["MS SubClass"].map(
        lambda valor: str(int(valor)) if valor == valor else valor
    )

    # Valor inconsistente detectado en el diagnóstico.
    X_limpio.loc[
        X_limpio["Garage Yr Blt"].eq(2207),
        "Garage Yr Blt"
    ] = float("nan")

    return X_limpio


def construir_preprocesador(X):
    """Construye el preprocesador sin ajustarlo."""
    columnas_numericas = X.select_dtypes(
        include="number"
    ).columns.tolist()

    columnas_categoricas = X.select_dtypes(
        exclude="number"
    ).columns.tolist()

    pipeline_numerico = Pipeline([
        (
            "imputacion",
            SimpleImputer(
                strategy="median",
                keep_empty_features=True
            )
        ),
        (
            "escalado",
            StandardScaler()
        )
    ])

    pipeline_categorico = Pipeline([
        (
            "imputacion",
            SimpleImputer(
                strategy="constant",
                fill_value="Desconocido",
                keep_empty_features=True
            )
        ),
        (
            "codificacion",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ])

    preprocesador = ColumnTransformer(
        transformers=[
            (
                "numericas",
                pipeline_numerico,
                columnas_numericas
            ),
            (
                "categoricas",
                pipeline_categorico,
                columnas_categoricas
            )
        ],
        remainder="drop"
    )

    return preprocesador


def preparar_y_guardar():
    """Ejecuta la preparación y guarda sus resultados."""
    X_train, X_test, y_train, y_test = cargar_y_separar()

    X_train_limpio = limpiar_datos(X_train)
    X_test_limpio = limpiar_datos(X_test)

    preprocesador = construir_preprocesador(X_train_limpio)

    train_array = preprocesador.fit_transform(X_train_limpio)
    test_array = preprocesador.transform(X_test_limpio)

    columnas = preprocesador.get_feature_names_out()

    train_features = pd.DataFrame(
        train_array,
        columns=columnas,
        index=X_train.index
    )

    test_features = pd.DataFrame(
        test_array,
        columns=columnas,
        index=X_test.index
    )

    train = train_features.copy()
    test = test_features.copy()

    train["target"] = y_train
    test["target"] = y_test

    # Recupera el orden original de las filas para el clustering.
    features_full = pd.concat(
        [train_features, test_features]
    ).sort_index()

    # Permite vincular cada fila del clustering con el CSV original.
    referencia_filas = pd.DataFrame({
        "indice_original": features_full.index
    })

    if train.isna().any().any() or test.isna().any().any():
        raise ValueError("Los conjuntos preparados contienen nulos.")

    if not train_features.columns.equals(test_features.columns):
        raise ValueError("Las columnas de entrenamiento y prueba difieren.")

    output_dir = DATA_PROCESSED
    output_dir.mkdir(parents=True, exist_ok=True)

    models_dir = MODELS
    models_dir.mkdir(parents=True, exist_ok=True)

    train.to_csv(output_dir / "train.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)
    features_full.to_csv(
        output_dir / "features_full.csv",
        index=False
    )
    referencia_filas.to_csv(
        output_dir / "referencia_filas.csv",
        index=False
    )

    joblib.dump(
        preprocesador,
        models_dir / "preprocesador.joblib"
    )

    return pd.DataFrame({
        "archivo": [
            "train.csv",
            "test.csv",
            "features_full.csv"
        ],
        "filas": [
            len(train),
            len(test),
            len(features_full)
        ],
        "columnas": [
            train.shape[1],
            test.shape[1],
            features_full.shape[1]
        ]
    })