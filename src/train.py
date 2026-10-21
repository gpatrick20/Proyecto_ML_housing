"""Funciones de entrenamiento y evaluación supervisada (Parte 2)."""

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline

from src.data_prep import cargar_y_separar, limpiar_datos
from src.data_prep import construir_preprocesador


RANDOM_STATE = 42
TOLERANCIA = 0.10        # error relativo máximo aceptado por el KPI de cobertura
META_COBERTURA = 0.70    # README sección 4.1: al menos 70% de predicciones dentro de la tolerancia
META_MEJORA_MAE = 0.15   # README sección 4.1: al menos 15% menos MAE que la referencia de la mediana
COLUMNAS_EXCLUIDAS = ["Sale Type", "Sale Condition"]  # se conocen solo al cerrar la venta


def preparar_entrada(X):
    """Aplica limpiar_datos y quita las variables posteriores a la estimación.

    No aprende estadísticas: es seguro usarla antes de la validación cruzada.
    """
    return limpiar_datos(X).drop(columns=COLUMNAS_EXCLUIDAS, errors="ignore")


def cargar_datos_crudos():
    """Devuelve X_train, X_test, y_train, y_test sin transformar y sin variables excluidas.

    Usa el mismo split de la Parte 1 (test_size=0.2, random_state=42).
    """
    X_train, X_test, y_train, y_test = cargar_y_separar()
    return preparar_entrada(X_train), preparar_entrada(X_test), y_train, y_test


def calcular_metricas(y_real, y_pred):
    """MAE y RMSE en USD, R2 y cobertura (% de predicciones con error relativo <= TOLERANCIA)."""
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_pred - y_real
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "R2": float(r2_score(y_real, y_pred)),
        "Cobertura_10": float(np.mean(np.abs(error) / np.abs(y_real) <= TOLERANCIA)),
    }


def crear_cv(n_splits=5):
    """Validación cruzada de 5 particiones con semilla fija."""
    return KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def construir_pipeline(regresor, X, log_objetivo=False):
    """Pipeline completo: preprocesador (se ajusta dentro de cada partición) + regresor.

    Con log_objetivo=True el modelo aprende log1p(SalePrice) y devuelve predicciones en USD.
    """
    if log_objetivo:
        regresor = TransformedTargetRegressor(
            regressor=regresor, func=np.log1p, inverse_func=np.expm1
        )
    return Pipeline([("prep", construir_preprocesador(X)), ("modelo", regresor)])


def construir_ridge(alpha=1.0):
    """Modelo lineal regularizado: baseline simple e interpretable."""
    return Ridge(alpha=alpha)


def referencia_mediana(X, y):
    """Referencia del KPI: predice la mediana de SalePrice del entrenamiento para todas las viviendas."""
    return DummyRegressor(strategy="median").fit(X, y)


def validacion_cruzada(estimador, X, y, cv=None):
    """Predicciones fuera de muestra con validación cruzada y sus métricas."""
    cv = crear_cv() if cv is None else cv
    predicciones = cross_val_predict(estimador, X, y, cv=cv)
    return calcular_metricas(y, predicciones), predicciones


def evaluar(estimador, X, y):
    """Métricas de un estimador ya ajustado sobre (X, y)."""
    return calcular_metricas(y, estimador.predict(X))


def evaluar_metas(metricas, mae_referencia):
    """Compara las métricas con las metas del README (cobertura y mejora del MAE)."""
    mejora = 1 - metricas["MAE"] / mae_referencia
    return {
        "Mejora_MAE": float(mejora),
        "Cumple_cobertura": bool(metricas["Cobertura_10"] >= META_COBERTURA),
        "Cumple_mejora": bool(mejora >= META_MEJORA_MAE),
    }
