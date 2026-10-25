"""Funciones de la Parte 3: segmentación no supervisada de viviendas.

La lógica reutilizable vive aquí; el notebook 03 la llama y comenta los resultados.
Todo es determinista (random_state=42).
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score

from src.utils import load_data, load_raw

SEMILLA = 42

# Describen la transacción y solo se conocen al cerrar la venta (README, sección 12.1).
# Se excluyen del clustering para que los segmentos describan la vivienda, no la venta.
PREFIJOS_EXCLUIDOS = ("categoricas__Sale Type", "categoricas__Sale Condition")


def cargar_features_clustering():
    """Devuelve (X, ref): X = features_full sin columnas de la venta; ref = índice original por fila."""
    features = load_data("features_full.csv")
    ref = load_data("referencia_filas.csv")["indice_original"]
    excluidas = [c for c in features.columns if c.startswith(PREFIJOS_EXCLUIDOS)]
    return features.drop(columns=excluidas), ref, excluidas


def cargar_originales_alineados(ref):
    """Datos originales (legibles) en el mismo orden que features_full."""
    raw = load_raw(keep_default_na=False, na_values=[""])
    return raw.loc[ref.to_numpy()].reset_index(drop=True)


def evaluar_k(Z, ks, n_boot=20, semilla=SEMILLA):
    """Métricas de K-Means para cada k: inercia, silhouette, Davies-Bouldin y estabilidad bootstrap.

    Estabilidad: ARI medio entre las etiquetas del modelo completo y las que asigna
    un modelo entrenado con una muestra bootstrap (1 = idénticas).
    """
    rng = np.random.RandomState(semilla)
    filas = []
    for k in ks:
        base = KMeans(k, n_init=20, random_state=semilla).fit(Z)
        aris = []
        for b in range(n_boot):
            idx = rng.choice(len(Z), len(Z), replace=True)
            kb = KMeans(k, n_init=10, random_state=b).fit(Z[idx])
            aris.append(adjusted_rand_score(base.labels_, kb.predict(Z)))
        filas.append({
            "k": k,
            "inercia": base.inertia_,
            "silhouette": silhouette_score(Z, base.labels_),
            "davies_bouldin": davies_bouldin_score(Z, base.labels_),
            "estabilidad_ARI": float(np.mean(aris)),
            "estabilidad_sd": float(np.std(aris)),
            "n_min_cluster": int(np.bincount(base.labels_).min()),
        })
    return pd.DataFrame(filas)


def nombrar_segmentos(df, etiqueta="cluster"):
    """Nombra cada cluster con reglas sobre características observables de la vivienda.

    Se nombra por reglas (no por número de cluster) para que los nombres no dependan
    del orden arbitrario de las etiquetas de K-Means. Falla si dos clusters reciben el mismo nombre.
    """
    nombres = {}
    for c, g in df.groupby(etiqueta):
        pct_duplex = g["Bldg Type"].isin(["Duplex", "2fmCon"]).mean()
        pct_2pisos = (g["House Style"] == "2Story").mean()
        pct_1piso = (g["House Style"] == "1Story").mean()
        anio = g["Year Built"].median()
        if pct_duplex >= 0.5:
            nombres[c] = "S1 Dúplex y multifamiliares"
        elif pct_2pisos >= 0.5:
            nombres[c] = "S5 Dos pisos amplias, construcción reciente"
        elif pct_1piso >= 0.9 and anio >= 1990:
            nombres[c] = "S4 Un piso de construcción reciente y mayor calidad"
        elif anio <= 1945:
            nombres[c] = "S2 Antiguas y compactas"
        else:
            nombres[c] = "S3 Un piso de tamaño medio, construcción de los años 60"
    if len(set(nombres.values())) != len(nombres):
        raise ValueError(f"Dos clusters recibieron el mismo nombre: {nombres}")
    return nombres


def perfilar_segmentos(df, columna="segmento"):
    """Tabla de perfil por segmento con variables observables y precio."""
    g = df.groupby(columna)
    perfil = g.agg(
        n=("SalePrice", "size"),
        precio_mediano=("SalePrice", "median"),
        precio_q1=("SalePrice", lambda s: s.quantile(0.25)),
        precio_q3=("SalePrice", lambda s: s.quantile(0.75)),
        superficie_habitable_pies2=("Gr Liv Area", "median"),
        calidad_general=("Overall Qual", "median"),
        anio_construccion=("Year Built", "median"),
        superficie_lote_pies2=("Lot Area", "median"),
        sotano=("Total Bsmt SF", "median"),
        autos_garaje=("Garage Cars", "median"),
        banos_completos=("Full Bath", "median"),
        pct_un_piso=("House Style", lambda s: (s == "1Story").mean()),
        pct_dos_pisos=("House Style", lambda s: (s == "2Story").mean()),
        pct_duplex_multifam=("Bldg Type", lambda s: s.isin(["Duplex", "2fmCon"]).mean()),
        pct_townhouse=("Bldg Type", lambda s: s.isin(["TwnhsE", "Twnhs"]).mean()),
        pct_zona_RM=("MS Zoning", lambda s: (s == "RM").mean()),
    )
    perfil["pct_del_total"] = perfil["n"] / perfil["n"].sum()
    return perfil


def intervalo_wilson(exitos, n, z=1.96):
    """Intervalo de confianza de Wilson (95 %) para una proporción."""
    if n == 0:
        return (np.nan, np.nan)
    p = exitos / n
    den = 1 + z**2 / n
    centro = (p + z**2 / (2 * n)) / den
    mitad = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return (centro - mitad, centro + mitad)


def error_por_segmento(df_prueba, col_real="SalePrice", col_pred="precio_predicho",
                       columna="segmento", tolerancia=0.10):
    """MAE, error relativo mediano y cobertura ±10 % por segmento (solo filas de prueba)."""
    d = df_prueba.copy()
    d["err_abs"] = (d[col_pred] - d[col_real]).abs()
    d["err_rel"] = d["err_abs"] / d[col_real]
    d["dentro"] = d["err_rel"] <= tolerancia
    filas = []
    for seg, g in d.groupby(columna):
        lo, hi = intervalo_wilson(int(g["dentro"].sum()), len(g))
        filas.append({
            columna: seg,
            "n_prueba": len(g),
            "MAE_USD": g["err_abs"].mean(),
            "error_relativo_mediano": g["err_rel"].median(),
            "cobertura_10": g["dentro"].mean(),
            "cobertura_IC95_inf": lo,
            "cobertura_IC95_sup": hi,
        })
    return pd.DataFrame(filas).set_index(columna)
