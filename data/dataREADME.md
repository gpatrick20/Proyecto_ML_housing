# Documentación de datos: Ames Iowa Housing

## 1. Fuente y origen
- **Dataset:** Ames Housing Dataset (Ames, Iowa, Estados Unidos), compilado originalmente por Dean De Cock (2011) como alternativa ampliada al dataset Boston Housing.
- **Provisión:** Entregado por Duoc UC para la asignatura de Machine Learning (`MLY1101`).
- **Archivo original:** `data/raw/Ames_Iowa_Housing_Dataset.csv` (conservado sin modificaciones).
- **Dimensiones:** 2.930 registros (viviendas vendidas entre 2006 y 2010) y 82 columnas.

## 2. Estructura de variables
- **Identificadores (2):** `Order` (número correlativo de observación) y `PID` (código catastral de la parcela). Excluidos del entrenamiento y de la predicción.
- **Variable objetivo (1):** `SalePrice` — precio final de venta de la propiedad en dólares estadounidenses (USD).
- **Variables predictoras (79):**
  - **20 continuas:** superficies y dimensiones en pies cuadrados (`Gr Liv Area`, `Total Bsmt SF`, `Lot Area`, `Garage Area`, `1st Flr SF`, `2nd Flr SF`, etc.) y frente de lote (`Lot Frontage`).
  - **14 discretas:** conteos de habitaciones, baños, cocinas, chimeneas, capacidad de garaje y años de construcción/remodelación (`Year Built`, `Year Remod/Add`, `Garage Yr Blt`, `Yr Sold`, `Mo Sold`).
  - **23 nominales:** ubicación y entorno (`Neighborhood`, `MS Zoning`, `Condition 1`, `Condition 2`), tipo de vivienda (`MS SubClass`, `Bldg Type`, `House Style`), materiales, techo, fundación, tipo de garaje (`Roof Style`, `Exterior 1st`, `Exterior 2nd`, `Foundation`, `Garage Type`) y condiciones de venta (`Sale Type`, `Sale Condition`).
  - **22 ordinales:** calificaciones de calidad y estado (`Overall Qual`, `Overall Cond`, `Exter Qual`, `Exter Cond`, `Bsmt Qual`, `Bsmt Cond`, `Kitchen Qual`, `Heating QC`, `Fireplace Qu`, `Garage Qual`, `Garage Cond`, `Pool QC`, etc.).

## 3. Notas de lectura, calidad y preprocesamiento
- **Códigos textuales `"NA"` y `"None"`:** En este dataset, `"NA"` significa ausencia legítima de la característica (por ejemplo, sin sótano, sin garaje, sin chimenea o sin piscina) y **no** un dato perdido. Solo las celdas verdaderamente vacías se interpretan como valores ausentes (`NaN`).
- **Variables posteriores a la venta:** `Sale Type` y `Sale Condition` se excluyen del modelado supervisado y del clustering porque solo se conocen al momento de cerrar la transacción.
- **Anomalías detectadas:** Se identificó una inconsistencia temporal (`Garage Yr Blt = 2207`), convertida a valor ausente (`NaN`) antes de imputar con la mediana de entrenamiento.
- **Codificación de tipos:** `MS SubClass` se trata como variable categórica textual para evitar interpretar sus códigos numéricos como distancias ordinales.

## 4. Archivos procesados (`data/processed/`)
Generados de forma reproducible mediante `notebooks/01_comprension_eda_preparacion.ipynb` (`src/data_prep.py`) y `notebooks/03_no_supervisado_y_conclusiones.ipynb` (`src/clustering.py`):

| Archivo | Filas | Columnas | Descripción |
|---|---:|---:|---|
| `train.csv` | 2.344 | 340 | Conjunto de entrenamiento (80 %, `random_state=42`): 339 predictores transformados + `target`. |
| `test.csv` | 586 | 340 | Conjunto de prueba (20 %, `random_state=42`): 339 predictores transformados + `target`. |
| `features_full.csv` | 2.930 | 339 | Predictores transformados de todas las viviendas (sin `target`), ordenados según el índice original. |
| `referencia_filas.csv` | 2.930 | 1 | Índice original correspondiente a cada fila de `features_full.csv`. |
| `clusters.csv` | 2.930 | 3 | Asignación de segmentos (`indice_original`, `cluster`, `segmento` de `S1` a `S5`). |

## 5. Licencia y restricciones de uso
- Dataset utilizado exclusivamente con fines académicos en el marco de la evaluación de Duoc UC.
- Su redistribución pública está sujeta a la verificación de las condiciones de uso del dataset original de Ames Housing (Dean De Cock, *Journal of Statistics Education*, Vol. 19, No. 3, 2011).