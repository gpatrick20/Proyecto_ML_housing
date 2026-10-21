# Proyecto de Machine Learning: Predicción de precios de viviendas

## Estado actual del proyecto

La Parte 1 está implementada: comprensión de datos, diagnóstico de calidad,
análisis exploratorio y preparación reproducible. Se verificó la ejecución
completa del notebook 01 y la carga de los archivos procesados.

El entrenamiento supervisado, la evaluación de modelos, el clustering y las
recomendaciones finales corresponden a las siguientes partes. Los objetivos
y metas de esas etapas son compromisos de trabajo, no resultados obtenidos.


## 1. Problema de negocio

Para este proyecto se plantea el caso de una inmobiliaria que necesita
apoyar la estimación del precio de venta de viviendas a partir de sus
características registradas.

El problema consiste en generar estimaciones consistentes y evaluar
cuánto se diferencian de los precios de venta observados. Estas
estimaciones servirían como referencia inicial para la valoración
de propiedades, sin reemplazar una tasación profesional.

Se utilizará el dataset Ames Iowa Housing y su variable objetivo
SalePrice. El proyecto comparará modelos de regresión y explorará
segmentos de viviendas para interpretar diferencias entre propiedades.

El caso de la inmobiliaria corresponde a un escenario de aplicación
propuesto por el equipo, no a una empresa que haya proporcionado
requerimientos o resultados operacionales.

## 2. Objetivo general

Desarrollar y evaluar una solución de Machine Learning que estime
el precio de venta de viviendas a partir de sus características,
comparando al menos dos modelos de regresión y analizando segmentos
de propiedades para apoyar la interpretación de los resultados.

## 3. Objetivos específicos

1. Antes del cierre de la Parte 1, analizar la calidad de los datos y
   documentar al menos cinco hallazgos exploratorios con su posible
   relevancia para el problema de negocio.

2. Antes del cierre de la Parte 1, implementar una preparación
   reproducible de los datos, documentando las decisiones de limpieza,
   transformación y separación entre entrenamiento y prueba.

3. Antes del cierre de la Parte 2, entrenar y comparar al menos dos
   modelos de regresión mediante validación cruzada sobre los datos
   de entrenamiento.

4. En la evaluación final de la Parte 2, comprobar si el modelo
   seleccionado alcanza las metas propuestas: al menos un 70% de
   predicciones con error relativo absoluto menor o igual al 10%
   y una reducción del MAE de al menos un 15% respecto de la
   referencia basada en la mediana del entrenamiento.

5. Antes del cierre de la Parte 3, aplicar al menos una técnica
   de aprendizaje no supervisado, caracterizar los segmentos
   identificados y justificar su utilidad para el caso de negocio.

6. Antes de la entrega final, formular al menos tres recomendaciones
   sustentadas en los resultados, indicando sus limitaciones.
## 4. Indicadores de éxito del proyecto

### 4.1. KPIs orientados al negocio

Los siguientes indicadores evalúan la utilidad de las estimaciones como
apoyo a la valoración de viviendas. No representan ahorros monetarios
ni mejoras operacionales demostradas.

| Indicador | Definición | Meta propuesta |
|---|---|---|
| Cobertura de estimaciones dentro de tolerancia | Porcentaje de viviendas cuyo precio predicho presenta un error absoluto relativo menor o igual al 10% respecto del precio observado | Alcanzar al menos un 70% |
| Mejora del error respecto de una estimación de referencia | Reducción porcentual del MAE frente a una referencia que predice la mediana de SalePrice del conjunto de entrenamiento para todas las viviendas | Reducir el MAE al menos un 15% |

Las metas son criterios iniciales propuestos por el equipo para este
proyecto académico, no requisitos de la pauta ni estándares del sector.
Se fijarán antes de comparar modelos y se informará su cumplimiento
sin modificarlas a partir de los resultados del conjunto de prueba.

### 4.2. Métricas técnicas

| Métrica | Propósito | Criterio de evaluación |
|---|---|---|
| MAE | Medir la magnitud promedio del error absoluto en las unidades originales de SalePrice | Menor que la referencia basada en la mediana |
| RMSE | Identificar el impacto de errores de predicción grandes | Comparar los modelos; un valor menor es preferible |
| R² | Evaluar el desempeño respecto de la referencia constante basada en la media | Comparar los modelos; un valor mayor es preferible |

La selección del modelo se realizará con validación cruzada sobre los
datos de entrenamiento. El conjunto de prueba se reservará para la
evaluación final.

### 4.3. Limitaciones de los indicadores

El dataset permite evaluar errores de predicción, pero no demostrar
reducciones en tiempos de venta, horas de trabajo, sobreprecios o costos.
Estos efectos requieren una evaluación operacional adicional.

## 5. Alcance y supuestos

- La variable objetivo es SalePrice. Su definición, moneda y unidades
  se documentarán mediante la fuente y el diccionario del dataset.
- El proyecto se limita a los registros y variables del dataset
  Ames Iowa Housing.
- No se asumirá que los resultados son directamente aplicables al
  mercado inmobiliario chileno ni a otros mercados.
- No se incorporarán fuentes externas en esta primera versión.
- Se revisará qué variables estarían disponibles al momento de
  realizar una estimación, para evitar utilizar información que
  solo se conocería después de la venta.
- Las predicciones se plantean como apoyo a la valoración, no como
  sustituto de una tasación profesional.
- No se evaluarán rentabilidad de inversión, tiempos de venta
  ni ahorros operacionales.

## 6. Metodología: CRISP-DM

El proyecto se organiza mediante las seis fases de CRISP-DM.
Cada fase tendrá actividades y evidencias asociadas dentro del repositorio.

La siguiente tabla describe la implementación prevista. Las evidencias
se completarán y actualizarán conforme avance el proyecto; su inclusión
en esta tabla no significa que las actividades ya estén terminadas.

### 6.1. Aplicación de las fases

| Fase | Aplicación en el proyecto | Evidencia prevista |
|---|---|---|
| 1. Comprensión del negocio | Definir el escenario de la inmobiliaria, la decisión que se busca apoyar, los objetivos, los KPIs y las limitaciones del uso de predicciones. | README.md: problema, objetivos, indicadores de éxito y alcance. |
| 2. Comprensión de los datos | Documentar la fuente y el significado de las variables. Revisar dimensiones, tipos, valores ausentes, duplicados, distribuciones y relaciones con SalePrice. | data/README.md y notebooks/01_comprension_eda_preparacion.ipynb: diagnóstico de calidad y EDA. |
| 3. Preparación de los datos | Definir las variables utilizables, separar entrenamiento y prueba y aplicar las transformaciones justificadas. Generar los archivos de datos preparados. | src/data_prep.py, src/utils.py, notebook 01 y data/processed/. |
| 4. Modelado | Entrenar y comparar al menos dos modelos de regresión. Aplicar una técnica no supervisada para identificar segmentos de viviendas. Justificar algoritmos y configuraciones. | notebooks/02_modelos_supervisados.ipynb, notebooks/03_no_supervisado_y_conclusiones.ipynb y models/. |
| 5. Evaluación | Comparar el desempeño mediante MAE, RMSE y R², comprobar las metas propuestas y analizar errores y limitaciones. Evaluar la coherencia y utilidad de los segmentos identificados. | Notebooks 02 y 03, tabla comparativa de resultados y conclusiones del README.md. |
| 6. Despliegue y comunicación | Entregar una solución académica reproducible, documentar su ejecución y comunicar resultados, recomendaciones y restricciones de uso. | README.md con instrucciones de reproducción, scripts, notebooks, modelos guardados y presentacion/. |

En este proyecto, la fase de despliegue se limitará a la entrega
reproducible y la comunicación de resultados. No se contempla publicar
un servicio de tasación ni implementar el modelo en una inmobiliaria.

### 6.2. Relación entre scripts y notebooks

La lógica reutilizable se organizará en scripts dentro de src/.
Los notebooks importarán esas funciones y mostrarán su ejecución,
sus resultados y las justificaciones correspondientes.

- src/data_prep.py: preparación de datos y generación de archivos
  procesados.
- src/utils.py: utilidades compartidas y función load_data() para
  entregar los conjuntos de entrenamiento y prueba.
- src/train.py: archivo reservado para las funciones de entrenamiento de la Parte 2; pendiente de implementación.
- src/predict.py: archivo reservado para la generación de predicciones con artefactos entrenados; pendiente de implementación.
- notebooks/: diagnóstico, análisis, explicación de decisiones,
  resultados y conclusiones.

La preparación implementada en los scripts no se duplicará en los
notebooks. Estos deberán permitir reproducir y comprender el proceso
mediante llamadas a las funciones correspondientes.

### 6.3. Controles metodológicos previstos

- Mantener los datos originales sin modificaciones en data/raw/.
- Documentar cada transformación y su justificación.
- Ajustar imputaciones, codificaciones, escalados y otras
  transformaciones aprendidas únicamente con los datos de entrenamiento.
- Durante la validación cruzada, ajustar las transformaciones
  aprendidas dentro de cada partición de entrenamiento.
- Seleccionar modelos e hiperparámetros sin utilizar el conjunto
  de prueba.
- Reservar el conjunto de prueba para la evaluación final de los modelos,
  sin utilizarlo para seleccionar algoritmos ni hiperparámetros.
- Reconocer que el EDA inicial utilizó el dataset completo: prueba no
  estuvo completamente aislada de la exploración.
- Utilizar random_state=42 en las operaciones aleatorias que admitan
  este parámetro.
- Mantener las rutas, dependencias e instrucciones necesarias para
  reproducir el proyecto.
- Diferenciar los resultados obtenidos de los beneficios de negocio
  que todavía no se han validado.

## 7. Evaluación del impacto ético

La evaluación ética considerará riesgos relacionados con los datos,
la calidad de las estimaciones y el uso de los resultados.

Los riesgos descritos a continuación son aspectos que se revisarán
durante el proyecto. No constituyen evidencia de que ya se haya
detectado discriminación, exposición de datos personales o daños
producidos por el modelo.

### 7.1. Riesgos y medidas de mitigación

| Riesgo a evaluar | Posible consecuencia | Medida prevista |
|---|---|---|
| Diferencias de desempeño entre grupos de viviendas | Un error promedio aceptable podría ocultar estimaciones deficientes en determinados barrios o rangos de precio. | Comparar errores por barrio y rango de precio cuando exista un número suficiente de registros; informar el tamaño de cada grupo y las limitaciones del análisis. |
| Uso de la ubicación como representación indirecta de factores sociales | Interpretar diferencias de precio como juicios sobre los habitantes o como fundamento para decisiones excluyentes. | Limitar las conclusiones a las propiedades y los datos observados; no atribuir características personales a sus habitantes ni usar el modelo para decidir sobre personas. |
| Aplicación fuera del contexto del dataset | Presentar estimaciones como válidas para otros mercados o momentos sin comprobar su desempeño. | Delimitar el alcance al dataset utilizado y exigir validación adicional antes de considerar su uso en otro contexto. |
| Uso automático de predicciones incorrectas | Tomar decisiones de compra, venta o valoración sin considerar errores y limitaciones. | Presentar el modelo como herramienta de apoyo, comunicar sus métricas y requerir revisión profesional para decisiones reales. |
| Privacidad y vinculación de registros | Facilitar la identificación o vinculación de propiedades mediante identificadores y fuentes externas. | Revisar la necesidad de los identificadores; no utilizarlos como predictores por defecto ni publicar cruces con información personal. |
| Tratamiento inadecuado de valores ausentes o extremos | Eliminar o representar incorrectamente viviendas poco frecuentes y reducir la calidad de sus estimaciones. | Consultar el diccionario de datos, justificar las transformaciones y no eliminar registros únicamente por ser inusuales. |
| Interpretación causal injustificada | Afirmar que una característica provoca un cambio de precio solo porque está asociada con él o resulta importante para el modelo. | Presentar los resultados como asociaciones predictivas, no como efectos causales demostrados. |
| Segmentación estigmatizante | Asignar etiquetas que descalifiquen barrios, propiedades o personas. | Nombrar los segmentos mediante características observables de las viviendas y aclarar que los grupos dependen de los datos y del método utilizado. |

### 7.2. Uso responsable y restricciones

Las predicciones no se presentarán como una tasación oficial ni como
una garantía del precio de venta de una propiedad.

El proyecto no se utilizará para decidir el acceso de personas a
créditos, seguros o vivienda.

Las recomendaciones deberán estar sustentadas en resultados
documentados. No se afirmarán ahorros monetarios, reducciones de
tiempos de venta ni beneficios operacionales que no hayan sido medidos.

Antes de publicar los datos se verificará su licencia y las condiciones
de redistribución. No se asumirá que un archivo puede redistribuirse
únicamente por haber sido entregado para una actividad académica.

### 7.3. Revisión al finalizar el proyecto

Esta evaluación se actualizará con las evidencias obtenidas durante
el análisis y el modelado. Se documentarán:

- Las diferencias de error detectadas entre grupos de viviendas.
- Los grupos con pocos registros y las limitaciones de su evaluación.
- Las decisiones de preparación que puedan afectar propiedades
  poco representadas.
- Las restricciones de interpretación y uso del modelo final.

La ausencia de diferencias evidentes en estas revisiones no se
interpretará como una garantía de ausencia de sesgos.

## 8. Fuentes de datos

Se utiliza el dataset Ames Iowa Housing, proporcionado por Duoc UC
para esta evaluación.

El archivo Ames_Iowa_Housing_Dataset.csv contiene 2.930 registros y
82 columnas, incluida la variable objetivo SalePrice.

El archivo original se conserva sin modificaciones en data/raw/.
La documentación complementaria se encuentra en
[data/README.md](data/README.md).

## 9. Calidad de datos y análisis exploratorio

### 9.1. Diagnóstico de calidad

El diagnóstico se desarrolla en
[notebooks/01_comprension_eda_preparacion.ipynb](notebooks/01_comprension_eda_preparacion.ipynb).

- El dataset contiene 2.930 registros y 82 columnas.
- No se encontraron filas completamente duplicadas ni repeticiones en
  los identificadores Order y PID.
- SalePrice no presenta valores ausentes, ceros ni valores negativos.
- La lectura predeterminada del CSV mezcla campos vacíos con códigos
  como "NA" y "None". En Mas Vnr Type, los 1.775 valores inicialmente
  interpretados como ausentes incluyen 1.752 textos "None" y 23 campos vacíos.
- Se identificaron cuatro viviendas con alertas de coherencia temporal.
  Las seis alertas se superponen entre esos registros; destaca
  Garage Yr Blt = 2207.
- La regla exploratoria de 1,5 × IQR marcó 127 observaciones en Lot Area,
  75 en Gr Liv Area, 123 en Total Bsmt SF y 137 en SalePrice.
  Los grupos pueden superponerse y no representan errores confirmados.

No se eliminaron variables únicamente por su porcentaje inicial de nulos
ni viviendas únicamente por superar los límites IQR.

### 9.2. Principales hallazgos

| Hallazgo | Evidencia observada | Relevancia para el negocio |
|---|---|---|
| Distribución asimétrica del precio | Media de 180.796,06 y mediana de 160.000; el 50 % central está entre 129.500 y 213.500. | Una sola referencia no representa toda la variedad de viviendas; conviene evaluar posteriormente los errores por rango de precio. |
| Asociación entre superficie habitable y precio | Correlación de Pearson de 0,707 entre Gr Liv Area y SalePrice. | La superficie ayuda a comparar propiedades, pero no resulta suficiente por sí sola. |
| Diferencias según calidad general | Mediana de 133.000 en Overall Qual = 5 y de 264.530,50 en Overall Qual = 8. | La calidad aporta contexto para comparar viviendas; los grupos pequeños requieren cautela. |
| Diferencias entre barrios | Medianas de 88.250 en MeadowV y 319.000 en StoneBr. | La ubicación debe interpretarse junto con las características de las viviendas y el tamaño de cada grupo. |
| Asociación entre año de construcción y precio | Correlación de Pearson de 0,558 entre Year Built y SalePrice. | La antigüedad debe complementarse con remodelaciones, calidad, superficie y barrio. |

Entre las variables seleccionadas para la matriz de correlaciones,
Overall Qual presenta la mayor correlación de Pearson con SalePrice (0,799).
También se observan relaciones elevadas entre predictores, como Garage Cars
y Garage Area (aproximadamente 0,89). No se eliminaron variables por
superar un umbral de correlación.

Estos resultados describen asociaciones, no efectos causales ni una
medida definitiva de importancia predictiva. Las diferencias entre
barrios no permiten atribuir características a sus habitantes.

### 9.3. Figuras del EDA

El notebook muestra y guarda las figuras en images/. Pueden consultarse
individualmente desde estos enlaces relativos:

- [Distribución del precio](images/eda_distribucion_saleprice.png).
- [Superficie habitable y precio](images/eda_superficie_precio.png).
- [Calidad general y precio](images/eda_calidad_precio.png).
- [Barrio y precio](images/eda_barrio_precio.png).
- [Año de construcción y precio](images/eda_anio_construccion_precio.png).
- [Matriz de correlaciones](images/eda_matriz_correlaciones.png).

## 10. Preparación de datos

La lógica reutilizable está implementada en src/data_prep.py y se ejecuta
una sola vez desde el notebook mediante preparar_y_guardar().

### 10.1. Decisiones implementadas

| Elemento | Tratamiento y justificación |
|---|---|
| Lectura | Preservar los textos "NA" y "None"; interpretar campos vacíos como ausentes. |
| Objetivo | Separar SalePrice y conservar su escala original; exportarlo con el nombre target. |
| Identificadores | Excluir Order y PID de los predictores por su función identificadora. |
| Partición | 80 % entrenamiento y 20 % prueba, con random_state=42. No se definieron estratos para el objetivo continuo. |
| MS SubClass | Convertir a categoría textual para no interpretar sus códigos como distancias numéricas. |
| Garage Yr Blt = 2207 | Convertir en ausente sin inventar el año correcto. |
| Otras alertas temporales | Conservarlas ante la falta de evidencia para identificar el campo incorrecto. |
| Valores extremos | Conservarlos cuando no exista evidencia suficiente de error. |
| Nulos numéricos | Imputar con la mediana aprendida en entrenamiento, evitando eliminar filas. |
| Nulos categóricos | Imputar con "Desconocido", distinguiendo los vacíos de los códigos originales y sin atribuir la categoría más frecuente. |
| Categorías | Aplicar OneHotEncoder con handle_unknown="ignore". Las categorías nuevas no detienen la transformación. |
| Escalas | Aplicar StandardScaler a las variables numéricas para disponer de una representación común. No todos los modelos necesitan este escalado. |

El preprocesador se ajusta únicamente con entrenamiento mediante
fit_transform(). Prueba se transforma con transform(), sin volver
a aprender medianas, categorías ni parámetros de escalado.

### 10.2. Archivos generados

| Archivo | Filas | Columnas | Contenido |
|---|---:|---:|---|
| data/processed/train.csv | 2.344 | 340 | 339 predictores transformados y target. |
| data/processed/test.csv | 586 | 340 | 339 predictores transformados y target. |
| data/processed/features_full.csv | 2.930 | 339 | Predictores de todos los registros, sin target. |
| data/processed/referencia_filas.csv | 2.930 | 1 | Índice original correspondiente a cada fila de features_full.csv. |
| models/preprocesador.joblib | No aplica | No aplica | Preprocesador ajustado con entrenamiento. |

features_full.csv combina los predictores transformados de entrenamiento
y prueba, ordenados por el índice original. No se vuelve a ajustar el
preprocesador con el conjunto completo.

### 10.3. Interfaz de carga

La función de src/utils.py devuelve los cuatro objetos acordados:

```python
from src.utils import load_data

X_train, X_test, y_train, y_test = load_data()
```

Dimensiones verificadas:

| Objeto | Filas | Variables |
|---|---:|---:|
| X_train | 2.344 | 339 |
| X_test | 586 | 339 |
| y_train | 2.344 | 1 |
| y_test | 586 | 1 |

y_train y y_test son Series con una sola variable objetivo.

La función también admite la carga de un archivo individual:

```python
features_full = load_data("features_full.csv")
referencia_filas = load_data("referencia_filas.csv")
```

La carga no realiza una nueva partición ni reajusta el preprocesador.
Se comprobaron la exclusión de target de los predictores, la igualdad
de columnas y su orden, la alineación entre predictores y objetivo,
y la ausencia de valores nulos.

Los índices de train y test se reconstruyen al leer sus CSV; no son los
identificadores originales ni demuestran por sí solos que no haya
viviendas compartidas entre conjuntos.

### 10.4. Limitaciones y continuidad metodológica

- El EDA se realizó sobre el dataset completo. Prueba no estuvo totalmente
  aislada de la exploración; no se utilizará para nuevas decisiones de
  preparación, selección de modelos o ajuste de hiperparámetros.
- La imputación por columna no garantiza coherencia entre variables
  relacionadas. Las alertas temporales no corregidas siguen siendo
  limitaciones del dataset preparado.
- Para validación cruzada, la Parte 2 deberá usar los datos anteriores a
  las transformaciones aprendidas y ajustar el preprocesamiento dentro
  de cada partición mediante un pipeline. Los CSV ya transformados no
  sustituyen ese ajuste. Pueden reutilizarse cargar_y_separar(),
  limpiar_datos() y construir_preprocesador() de src/data_prep.py.
- La revisión de disponibilidad de variables al momento de la estimación,
  planteada en el alcance, debe confirmarse antes de cerrar el modelado.
- La correspondencia de referencia_filas.csv permite recuperar las
  características originales para describir segmentos. Debe preservarse
  el orden de las filas al vincular archivos.

## 11. Reproducción de la Parte 1

### 11.1. Preparación del entorno

Desde una terminal en la raíz del proyecto:

```bash
python -m venv .venv
```

Activación en Git Bash para Windows:

```bash
source .venv/Scripts/activate
```

En PowerShell, como alternativa:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instalar las dependencias fijadas en el repositorio:

```bash
python -m pip install -r requirements.txt
```

### 11.2. Ejecución

1. Verificar la presencia de data/raw/Ames_Iowa_Housing_Dataset.csv.
2. Abrir el proyecto en VS Code.
3. Abrir notebooks/01_comprension_eda_preparacion.ipynb.
4. Seleccionar el kernel del entorno .venv.
5. Reiniciar el kernel y ejecutar todas las celdas en orden.
6. Guardar el notebook con las salidas visibles.

La preparación crea las carpetas de salida necesarias, guarda figuras,
genera o reemplaza los CSV procesados y persiste el preprocesador.
El CSV original no se modifica.

Las importaciones de src requieren que la raíz del proyecto esté
incluida en la ruta de importación. El notebook configura esta ruta
automáticamente cuando se abre desde la raíz o desde notebooks/.

### 11.3. Evidencia de ejecución y alcance de la verificación

Se verificó localmente la ejecución completa del notebook revisado y la
carga de los archivos generados. Esta comprobación no equivale todavía
a la prueba integral del proyecto en un entorno limpio, prevista para
la entrega final después de completar las Partes 2 y 3.

## 12. Entrega y trabajo pendiente

### Parte 1: implementada

- Definición del problema, objetivos y KPIs.
- Metodología CRISP-DM y evaluación ética inicial.
- Diagnóstico de calidad y cinco hallazgos exploratorios.
- Preparación reproducible, exportación y persistencia del preprocesador.
- Interfaz de carga y comprobaciones de los datos preparados.
- Resumen de resultados e instrucciones de ejecución en este documento.

La documentación de fuente y licencia debe mantenerse en data/README.md.
La redistribución del CSV está sujeta a la comprobación de sus condiciones
de uso; esta entrega no afirma una licencia que no haya sido verificada.


##PARTE 2:

## Modelado supervisado

**Modelos.** Se compararon dos modelos de regresión de familias distintas, con el mismo split (80 %/20 %,
`random_state=42`), validación cruzada de 5 particiones y las mismas métricas:

- **Ridge** (lineal regularizado, objetivo `log1p(SalePrice)`): baseline simple e interpretable.
- **Gradient boosting** (`HistGradientBoostingRegressor`): ensamble de árboles que captura relaciones no lineales e interacciones.

**Preparación.** El preprocesamiento (imputación, escalado y codificación) forma parte del Pipeline y se ajusta
dentro de cada partición de entrenamiento. Se excluyeron `Sale Type` y `Sale Condition` por conocerse solo al cerrar la venta.

**Configuración.** Ajuste con `GridSearchCV` (criterio: MAE en validación cruzada, sin usar la prueba).
Ridge ajustado: `{"modelo__regressor__alpha": 1}`.
Boosting ajustado: `{"modelo__l2_regularization": 1.0, "modelo__learning_rate": 0.03, "modelo__max_depth": null, "modelo__max_iter": 400}`.

**Resultados.**

| Modelo | MAE CV (USD) | MAE prueba (USD) | RMSE prueba (USD) | R² prueba | Cobertura ±10 % prueba |
|---|---|---|---|---|---|
| Referencia (mediana) | 54,123 | 63,838 | 94,323 | -0.110 | 18.6% |
| Ridge base | 16,064 | 14,940 | 32,246 | 0.870 | 75.1% |
| Ridge ajustado | 16,064 | 14,940 | 32,246 | 0.870 | 75.1% |
| Boosting base | 14,917 | 15,624 | 27,027 | 0.909 | 74.2% |
| Boosting ajustado | 14,635 | 15,350 | 27,285 | 0.907 | 74.7% |

**Modelo seleccionado:** Boosting ajustado (menor MAE en validación cruzada: 14,635 USD).
En prueba obtuvo un MAE de 15,350 USD, una mejora de 76.0% respecto de la referencia
de la mediana (meta >= 15%: cumple) y una cobertura ±10 % de 74.7%
(meta >= 70%: cumple). Las metas no se modificaron a partir de estos resultados.

**Interpretación inicial.** Las métricas describen el desempeño sobre las 586 viviendas de prueba del dataset Ames;
no demuestran beneficios operacionales ni validez en otros mercados. Los errores se revisaron por barrio y rango de precio
(`images/modelos_error_barrio.png`); los grupos con pocas viviendas no permiten conclusiones firmes. La importancia de variables
expresa asociaciones predictivas, no efectos causales. Las recomendaciones corresponden a la Parte 3.

**Artefactos.** `models/best_model.joblib` (Pipeline completo, recibe datos crudos), `models/results_supervised.csv`,
`src/train.py`, `src/predict.py` e imágenes `images/modelos_*.png`.


### Parte 3: pendientes


