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
from sklearn.base import clone
from scipy.stats import loguniform
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import KFold, cross_val_predict, learning_curve
from sklearn.pipeline import Pipeline

from src.data_prep import cargar_y_separar, limpiar_datos
from src.data_prep import construir_preprocesador, recortar_z
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


def construir_pipeline(regresor, X, log_objetivo=False, recorte_z=None):
    """Pipeline completo: preprocesador (se ajusta dentro de cada partición) + regresor.

    Con log_objetivo=True el modelo aprende log1p(SalePrice) y devuelve predicciones en USD.
    Con recorte_z=<número> las variables numéricas estandarizadas se recortan a [-z, z]
    (ver construir_preprocesador). Por defecto no hay recorte (comportamiento original).
    """
    if log_objetivo:
        regresor = TransformedTargetRegressor(
            regressor=regresor, func=np.log1p, inverse_func=np.expm1
        )
    return Pipeline([
        ("prep", construir_preprocesador(X, recorte_z=recorte_z)),
        ("modelo", regresor),
    ])


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



# ---------------------------------------------------------------------------
# Persona 1: optimización ampliada de hiperparámetros y evaluación de generalización
# ---------------------------------------------------------------------------

SIN_RECORTE = 1e6  # recorte tan amplio que equivale a no recortar (permite buscar "z" en la grilla)
ALPHAS_RIDGE = [0.1, 0.3, 1, 3, 10, 30, 100, 300]
RECORTES_Z = [SIN_RECORTE, 5.0, 4.0, 3.0, 2.5, 2.0]

# Ridge robusto: se busca alpha y el nivel de recorte z a la vez, solo con datos de entrenamiento.
GRILLA_RIDGE_ROBUSTO = {
    "modelo__regressor__alpha": ALPHAS_RIDGE,
    "prep__numericas__recorte__kw_args": [recortar_z(z) for z in RECORTES_Z],
}

# Boosting: espacio ampliado para la búsqueda aleatoria. learning_rate se muestrea en escala logarítmica.
ESPACIO_BOOSTING = {
    "modelo__learning_rate": loguniform(0.01, 0.2),
    "modelo__max_iter": [500, 800, 1200],            # tope: el early stopping decide cuántas se usan
    "modelo__max_depth": [3, 5, 8, None],
    "modelo__max_leaf_nodes": [15, 31, 63],
    "modelo__min_samples_leaf": [5, 10, 20, 40],
    "modelo__l2_regularization": [0.0, 1.0, 5.0, 10.0],
    "modelo__loss": ["squared_error", "absolute_error"],  # absolute_error optimiza directamente el MAE
}


def construir_ridge_robusto(X, alpha=1.0, recorte_z=SIN_RECORTE):
    """Ridge con objetivo log1p y recorte de variables numéricas estandarizadas."""
    return construir_pipeline(construir_ridge(alpha), X, log_objetivo=True, recorte_z=recorte_z)


def construir_boosting_es(**parametros):
    """HistGradientBoosting con parada temprana (validación interna del 10 %, paciencia de 30 iteraciones).

    max_iter pasa a ser un tope y no un valor fijo; la parada temprana hace viable explorar muchas configuraciones.
    """
    base = dict(early_stopping=True, validation_fraction=0.1, n_iter_no_change=30)
    base.update(parametros)
    return HistGradientBoostingRegressor(random_state=RANDOM_STATE, **base)


def buscar_aleatoria(estimador, espacio, X, y, n_iter=30, cv=None):
    """RandomizedSearchCV que minimiza el MAE en validación cruzada (solo con datos de entrenamiento)."""
    cv = crear_cv() if cv is None else cv
    busqueda = RandomizedSearchCV(
        estimador, espacio, n_iter=n_iter, cv=cv, scoring="neg_mean_absolute_error",
        n_jobs=-1, refit=True, random_state=RANDOM_STATE, return_train_score=True,
    )
    return busqueda.fit(X, y)


def buscar_grilla_con_train(estimador, grilla, X, y, cv=None):
    """GridSearchCV como buscar_hiperparametros, pero guardando también el error de entrenamiento."""
    cv = crear_cv() if cv is None else cv
    busqueda = GridSearchCV(
        estimador, grilla, cv=cv, scoring="neg_mean_absolute_error",
        n_jobs=-1, refit=True, return_train_score=True,
    )
    return busqueda.fit(X, y)


def tabla_busqueda(busqueda):
    """Resultados de una búsqueda como tabla ordenada por MAE de validación cruzada (USD)."""
    r = pd.DataFrame(busqueda.cv_results_)
    tabla = pd.DataFrame(list(r["params"]))
    tabla.columns = [c.split("__")[-1] for c in tabla.columns]
    if "kw_args" in tabla.columns:
        tabla["recorte_z"] = tabla.pop("kw_args").map(lambda d: d["a_max"])
    if "max_depth" in tabla.columns:
        tabla["max_depth"] = tabla["max_depth"].map(lambda v: "sin límite" if pd.isna(v) else str(int(v)))
    tabla["MAE_CV"] = -r["mean_test_score"].to_numpy()
    tabla["MAE_CV_desv"] = r["std_test_score"].to_numpy()
    tabla["MAE_entrenamiento"] = -r["mean_train_score"].to_numpy()
    tabla["segundos_ajuste"] = r["mean_fit_time"].to_numpy()
    return tabla.sort_values("MAE_CV").reset_index(drop=True)


def tabla_sensibilidad(tabla, parametros, bins=4):
    """MAE medio y mínimo de la búsqueda según cada valor de cada hiperparámetro.

    Los hiperparámetros continuos se agrupan en cuartiles. Sirve para ver cuánto pesa cada uno.
    """
    filas = []
    for par in parametros:
        valores = tabla[par]
        if valores.dtype.kind == "f" and valores.nunique() > 8:
            grupos = pd.qcut(valores, bins, duplicates="drop").astype(str)
        else:
            grupos = valores.astype(str)
        agg = tabla.groupby(grupos)["MAE_CV"].agg(["size", "mean", "min"])
        for valor, f in agg.iterrows():
            filas.append({"hiperparametro": par, "valor": valor, "n": int(f["size"]),
                          "MAE_CV_medio": f["mean"], "MAE_CV_minimo": f["min"]})
    return pd.DataFrame(filas)


def metricas_por_fold(estimador, X, y, cv=None):
    """Ajusta el estimador en cada partición y devuelve las métricas de validación y de entrenamiento por fold."""
    cv = crear_cv() if cv is None else cv
    filas = []
    for k, (idx_entr, idx_val) in enumerate(cv.split(X), start=1):
        modelo = clone(estimador).fit(X.iloc[idx_entr], y.iloc[idx_entr])
        m_val = calcular_metricas(y.iloc[idx_val], modelo.predict(X.iloc[idx_val]))
        m_ent = calcular_metricas(y.iloc[idx_entr], modelo.predict(X.iloc[idx_entr]))
        filas.append({"fold": k, **m_val,
                      "MAE_entrenamiento": m_ent["MAE"], "R2_entrenamiento": m_ent["R2"]})
    return pd.DataFrame(filas)


def resumir_folds(tabla_folds):
    """Media y desviación estándar (ddof=1) de cada métrica de la tabla por fold."""
    resumen = {}
    for col in tabla_folds.columns.drop("fold"):
        resumen[f"{col}_media"] = float(tabla_folds[col].mean())
        resumen[f"{col}_desv"] = float(tabla_folds[col].std(ddof=1))
    return resumen


def cv_anidada(estimador, espacio, X, y, n_iter=None, splits_internos=3, cv_externo=None):
    """Validación cruzada anidada: la búsqueda de hiperparámetros se repite dentro de cada fold externo.

    El fold externo de validación nunca interviene en la elección de hiperparámetros, así que su MAE no es
    optimista. Con n_iter se usa RandomizedSearchCV; sin n_iter, GridSearchCV.
    """
    cv_externo = crear_cv() if cv_externo is None else cv_externo
    cv_interno = KFold(n_splits=splits_internos, shuffle=True, random_state=RANDOM_STATE)
    filas = []
    for k, (idx_entr, idx_val) in enumerate(cv_externo.split(X), start=1):
        if n_iter is None:
            busqueda = GridSearchCV(estimador, espacio, cv=cv_interno,
                                    scoring="neg_mean_absolute_error", n_jobs=-1)
        else:
            busqueda = RandomizedSearchCV(estimador, espacio, n_iter=n_iter, cv=cv_interno,
                                          scoring="neg_mean_absolute_error", n_jobs=-1,
                                          random_state=RANDOM_STATE)
        busqueda.fit(X.iloc[idx_entr], y.iloc[idx_entr])
        m = calcular_metricas(y.iloc[idx_val], busqueda.predict(X.iloc[idx_val]))
        filas.append({"fold_externo": k, "MAE_externo": m["MAE"], "RMSE_externo": m["RMSE"],
                      "R2_externo": m["R2"], "Cobertura_10_externo": m["Cobertura_10"],
                      "MAE_interno_optimista": -busqueda.best_score_,
                      "mejores_hiperparametros": json.dumps(busqueda.best_params_, default=str, ensure_ascii=False)})
    return pd.DataFrame(filas)


def curva_aprendizaje(estimador, X, y, cv=None, tamanos=(0.2, 0.4, 0.6, 0.8, 1.0)):
    """MAE de entrenamiento y de validación según la cantidad de datos de entrenamiento."""
    cv = crear_cv() if cv is None else cv
    n, entr, val = learning_curve(
        estimador, X, y, cv=cv, train_sizes=list(tamanos),
        scoring="neg_mean_absolute_error", n_jobs=-1, random_state=RANDOM_STATE, shuffle=True,
    )
    return pd.DataFrame({
        "n_entrenamiento": n,
        "MAE_entrenamiento": -entr.mean(axis=1), "MAE_entrenamiento_desv": entr.std(axis=1, ddof=1),
        "MAE_validacion": -val.mean(axis=1), "MAE_validacion_desv": val.std(axis=1, ddof=1),
    })


def guardar_json(objeto, ruta):
    """Guarda un diccionario como JSON legible (UTF-8)."""
    from pathlib import Path
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(objeto, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    return ruta
