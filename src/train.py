"""Funciones de entrenamiento y evaluación supervisada (Parte 2)."""

import numpy as np
from sklearn.metrics import r2_score

from src.data_prep import cargar_y_separar, limpiar_datos


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
