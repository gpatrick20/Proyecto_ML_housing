# Datos del proyecto

## Fuente

El dataset fue proporcionado por Duoc UC para la Evaluación Parcial
N°2 de la asignatura Machine Learning (MLY1101).

## Descripción

| Característica | Detalle |
|---|---|
| Archivo | Ames_Iowa_Housing_Dataset.csv |
| Formato | CSV |
| Registros | 2.930 |
| Columnas | 82 |
| Variable objetivo | SalePrice |
| Problema | Predicción de precios de viviendas mediante regresión |

## Organización

- data/raw/: contiene el archivo original, que se conserva sin cambios.
- data/interim/: contiene archivos intermedios, si se requieren.
- data/processed/: contiene los datos preparados para las siguientes
  etapas del proyecto.

La preparación se implementará en src/data_prep.py y se documentará
en notebooks/01_comprension_eda_preparacion.ipynb.