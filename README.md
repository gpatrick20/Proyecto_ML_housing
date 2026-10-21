# Proyecto de Machine Learning: Predicción de precios de viviendas

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

El proyecto se organizará mediante las seis fases de CRISP-DM.
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
- src/train.py: funciones de entrenamiento del modelado supervisado.
- src/predict.py: generación de predicciones a partir de los artefactos
  entrenados.
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
- Reservar el conjunto de prueba para la evaluación final.
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