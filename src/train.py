"""Funciones de entrenamiento y evaluación supervisada (Parte 2)."""

import joblib
import json
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline

from src.data_prep import cargar_y_separar, limpiar_datos
from src.data_prep import construir_preprocesador
from src.utils import MODELS


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


def construir_boosting(**parametros):
    """Ensamble de árboles con gradient boosting (HistGradientBoostingRegressor)."""
    return HistGradientBoostingRegressor(random_state=RANDOM_STATE, **parametros)


# Los nombres siguen la convención de Pipeline: <paso>__<parametro>.
GRILLA_RIDGE = {
    "modelo__regressor__alpha": [0.1, 0.3, 1, 3, 10, 30, 100],
}

GRILLA_BOOSTING = {
    "modelo__learning_rate": [0.03, 0.05, 0.1],
    "modelo__max_depth": [3, 6, None],
    "modelo__max_iter": [200, 400],
    "modelo__l2_regularization": [0.0, 1.0],
}


def buscar_hiperparametros(estimador, grilla, X, y, cv=None):
    """GridSearchCV que minimiza el MAE en validación cruzada (solo con datos de entrenamiento)."""
    cv = crear_cv() if cv is None else cv
    busqueda = GridSearchCV(
        estimador, grilla, cv=cv, scoring="neg_mean_absolute_error",
        n_jobs=-1, refit=True,
    )
    return busqueda.fit(X, y)


def fila_resultado(nombre, familia, hiperparametros, m_cv, m_prueba,
                   mae_referencia_prueba=None, seleccionado=False):
    """Una fila de la tabla comparativa (validación cruzada y prueba)."""
    fila = {
        "modelo": nombre,
        "familia": familia,
        "hiperparametros": json.dumps(hiperparametros, default=str, ensure_ascii=False),
        "MAE_CV": m_cv["MAE"], "RMSE_CV": m_cv["RMSE"],
        "R2_CV": m_cv["R2"], "Cobertura10_CV": m_cv["Cobertura_10"],
        "MAE_prueba": m_prueba["MAE"], "RMSE_prueba": m_prueba["RMSE"],
        "R2_prueba": m_prueba["R2"], "Cobertura10_prueba": m_prueba["Cobertura_10"],
        "Mejora_MAE_prueba": None, "Cumple_meta_cobertura": None,
        "Cumple_meta_mejora": None, "seleccionado": bool(seleccionado),
    }
    if mae_referencia_prueba is not None:
        metas = evaluar_metas(m_prueba, mae_referencia_prueba)
        fila["Mejora_MAE_prueba"] = metas["Mejora_MAE"]
        fila["Cumple_meta_cobertura"] = metas["Cumple_cobertura"]
        fila["Cumple_meta_mejora"] = metas["Cumple_mejora"]
    return fila


def importancia_permutacion(estimador, X, y, n_repeats=10):
    """Importancia por permutación sobre las variables originales (aumento del MAE en USD)."""
    resultado = permutation_importance(
        estimador, X, y, scoring="neg_mean_absolute_error",
        n_repeats=n_repeats, random_state=RANDOM_STATE,
    )
    return (
        pd.DataFrame({
            "Variable": X.columns,
            "Aumento_MAE": resultado.importances_mean,
            "Desv": resultado.importances_std,
        })
        .sort_values("Aumento_MAE", ascending=False)
        .reset_index(drop=True)
    )


def coeficientes_ridge(pipeline):
    """Coeficientes del Ridge (variables estandarizadas, escala log de SalePrice)."""
    nombres = pipeline.named_steps["prep"].get_feature_names_out()
    coeficientes = pipeline.named_steps["modelo"].regressor_.coef_
    tabla = pd.DataFrame({
        "Variable": [n.split("__", 1)[-1] for n in nombres],
        "Coeficiente": coeficientes,
    })
    return tabla.reindex(tabla["Coeficiente"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def rangos_precio(y_train, y, cortes=(0.25, 0.5, 0.75)):
    """Asigna cada precio a un rango definido con los cuartiles del entrenamiento."""
    q = np.quantile(y_train, cortes)
    etiquetas = [
        f"Q1: hasta {q[0]:,.0f}",
        f"Q2: {q[0]:,.0f}-{q[1]:,.0f}",
        f"Q3: {q[1]:,.0f}-{q[2]:,.0f}",
        f"Q4: más de {q[2]:,.0f}",
    ]
    return pd.cut(y, bins=[-np.inf, *q, np.inf], labels=etiquetas)


def error_por_grupo(resultados, columna, min_n=10):
    """Error por grupo. Requiere las columnas y_real, y_pred y la columna de agrupación."""
    r = resultados.assign(error_abs=(resultados["y_pred"] - resultados["y_real"]).abs())
    r["error_rel"] = r["error_abs"] / r["y_real"]
    tabla = (
        r.groupby(columna, observed=True)
        .agg(n=("y_real", "size"), MAE=("error_abs", "mean"),
             Error_rel_mediano=("error_rel", "median"),
             Cobertura_10=("error_rel", lambda s: (s <= TOLERANCIA).mean()))
        .reset_index()
    )
    tabla["Muestra_suficiente"] = tabla["n"] >= min_n
    return tabla.sort_values("n", ascending=False).reset_index(drop=True)


def tabla_markdown(df):
    """Convierte un DataFrame ya formateado a tabla Markdown (para pegar en el README)."""
    columnas = list(df.columns)
    lineas = ["| " + " | ".join(columnas) + " |", "|" + "|".join(["---"] * len(columnas)) + "|"]
    for _, fila in df.iterrows():
        lineas.append("| " + " | ".join(str(v) for v in fila) + " |")
    return "\n".join(lineas)


def guardar_modelo(pipeline, ruta=None):
    """Guarda el Pipeline completo (preprocesamiento + modelo). Recibe datos crudos al predecir."""
    ruta = MODELS / "best_model.joblib" if ruta is None else ruta
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, ruta)
    return ruta


def guardar_resultados(tabla, ruta=None):
    """Guarda la tabla comparativa de modelos (la consume la Parte 3)."""
    ruta = MODELS / "results_supervised.csv" if ruta is None else ruta
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(ruta, index=False)
    return ruta
