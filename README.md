# PRACTICA1
Proyecto de Machine Learning aplicado a la prediccion de impago de prestamos (LendingClub dataset, 2007-2017). El objetivo es construir un modelo que, dado un conjunto de variables de un prestamo, prediga si el prestatario pagara o entrara en impago (Charged Off).

A lo largo del trabajo se han evaluados tres modelos sobre una misma base de datos variables_withExperts.xlsx .

Para dicho estudio se ha entrenado un pipeline end- to-end de Machile Learning, aplicando el modelo seguido en clase, pero con las siguientes modificaciones: 

- CLASE PREPROCESAMIENTO

Para la imputación de missings se ha aplicado la herramienta SimpleImputer con el parámetro most frequent, de forma que los valores faltantes quedan completos con el valor más repetido. También se ha utilizado KNNImputer, que permite rellenar los valores faltantes con los 5 vecinos más parecidos.

En el procesamiento de variables categóricas se ha sustituido el uso de OneHotEncoder por CountFrequencyEncoder. Mediante esta técnica, cada variable categórica queda sustituida por su frecuencia o su conteo dentro de los datos. De esta forma, las variables categóricas se transforman en variables numéricas.

La gestión de las variables numéricas se ha realizado mediante la herramienta MinMaxScaler. Utiliza el valor máximo y el valor mínimo para escalar las variables en un rango entre 0 y 1. El mayor inconveniente de este método es que es muy sensible a outliers. 

Además de procesar las variables ya incluidas en el dataset que estamos estudiando, se han creado combinaciones de variables ya existentes para generar nuevas. Así, se añaden a la base de datos las variables: range_fico, mean_FICO, fico_category, ratio_recent_delinquency.

-CLASE FILTRADO

Para la realización del filtrado, se ha aplicado SelectFromModel con RandomForestClassifier. De esta forma, se seleccionan automáticamente las variables más importantes del dataset.

- ENTRENAMIENTO DE MODELOS Y ANÁLISIS DE RESULTADOS

 En la tabla presentada a continuacion se muestran los diferentes valores de las métricas evaluadas en cada modelo, incluyendo el modelo entrenado en clase, el "modelo base". 
 
| Modelo             | Accuracy | Precision | Recall | F1-score | PR-AUC | ROC-AUC |
|--------------------|----------|---------- |--------|----------|--------|---------|
| Modelo Base        | 0.8004   | 0.5130    | 0.0198 | 0.0381   | —      | 0.6781  |
| Gradient Boosting  | 0.8010   | 0.5116    | 0.0941 | 0.1589   | 0.3664 | —       |
| SVM                | 0.7166   | 0.2647    | 0.2352 | 0.2491   | 0.1575 | —       |
| MLP                | 0.8006   | 0.5046    | 0.1228 | 0.1976   | 0.3709 | —       |

#### Gradient Boosting vs Modelo Base:
El modelo de Gradient Boosting presenta prácticamente la misma accuracy (≈0.80) y precisión  (≈0.51) que el modelo base. Sin embargo, incrementando de forma significativa el recall (de 0.0198 a 0.0941). Esto implica que es capaz de detectar más impagos, aunque sigue siendo un modelo conservador, dado que Gradient Boosting solo es capaz de predecir correctamnete el 9.41% de los impagos. En conjunto, supone una mejora directa sobre el modelo base, ya que aumenta la capacidad de detección sin empeorar el rendimiento global.

#### SVM vs Modelo Base:
El modelo de Máquinas de Soporte Vectorial (SVM) presenta un comportamiento diferente a la comparativa realizada anteriormente. Reduce la accuracy y la precisión respecto al modelo base, pero mejora de forma muy notable el recall (hasta 0.2352). Esto significa que detecta muchos más impagos, aunque en consecuencia, genera un mayor número de falsos positivos. En comparación con el modelo base, el SVM es más agresivo y útil para detectar riesgo, pero menos fiable en sus predicciones.

#### MLP vs Modelo Base:
La red neuronal (MLP) se presenta como la mejor opcion frente al modelo base dado que consigue un buen equilibrio entre accuracy y precision. Mientras que estos dos valores se mantienen muy similares en los dos modelos, la red neuronal mejora de forma notable el recall (de 0.0198 a 0.1228). Esto significa que detecta mejor los casos de impago sin perder demasiada calidad en las predicciones. 
