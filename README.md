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