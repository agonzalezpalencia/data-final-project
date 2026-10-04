# Proyecto Urgencias

## 1. Contexto y objetivos

## 2. Dataset (MIMIC-IV-ED Demo)

Dataset: MIMIC-IV-ED Demo
Número de registros brutos: 222
Campos: 
    1. subject_id	
    2. stay_id	
    3. temperature	
    4. heartrate	
    5. resprate	
    6. o2sat	
    7. sbp	
    8. dbp	
    9. pain	
    10. acuity	
    11. chiefcomplaint

### 2.1. Variable objetivo

`acuity`


## 3. Análisis del dataset

### 3.1. Matriz correlación


## 4. Limpieza del dataset

### 4.1. Descartes y transformaciones

- Limpiar valores nulos de todos los campos y eliminar esos registros.
- Eliminar las columnas `subject_id` y `stay_id` por no ser datos relevantes para el entrenamiento del modelo.
- Eliminar la columna `chiefcomplaint` por la inconsistencia de su contenido y la dificultad de normalizar sus valores.
- Binarizar `acuity` a:
  - 0: urgente (acuity 1-2)
  - 1: no urgente (acuity 3-5)
- Reemplazar por 0 en columna `pain` los valores UA, unable, uta, ett, o.

### 4.2. Resultado limpieza

Número de registros tras limpieza: 190
Campos finales del dataset: 
    1. temperature	
    2. heartrate	
    3. resprate	
    4. o2sat	
    5. sbp	
    6. dbp	
    7. pain	
    8. acuity	

### 4.3. Partición train/test



## 5. Los modelos

> Todos los modelos tendrán una semilla con valor de 42.

Hemos seleccionado 3 modelos para este dataset. 

El primero sería un modelo de clasificación binaria de Tensorflow la API de `Keras.Sequential` para poder ver comparaciones entre las librerias de Tensorflow y Scikit-Learn

El segundo modelo que hemos integrado es un `RandomForestClassifier` de Scikit-Learn, que se basa en una estructura de árboles de decisión para ofrecer predicciones y funciona muy bien con datasets de tamaños no muy elevados, como del que disponemos.

### 5.1. Pruebas

Utilizamos Jupyter Notebook para hacer pruebas sin necesidad de ejecutar el flujo de Airflow descrito posteriormente, además que el sistema de celdas nos permitió cambiar parámetros del modelo y volver a lanzarlo sin necesidad de modificar secciones como la lectura o preparado del dataset o las gráficas que se ven posteriormente.

Para LogisticRegression se hacen pruebas con varias configuraciones de hiperparámetros del modelo para seleccionar la mejor entre las 42 diferentes.

![alt text](img/image-4.png)

Estos son los datos concretos de las 10 mejores configuraciones:

![alt text](img/image-5.png)

### 5.2. Creación

El modelo de Tensorflow cuenta con con 4 capas:
- Capa de entrada con una forma de 7x1.
- Una capa de normalización ajustada a la porción de entrenamiento de X.
- Una capa de procesamiento de 32 neuronas con activación `relu`.
- La capa de salida de 1 neurona con activación `sigmoid` para la clasificación binaria.
Se ha usado para compilarlo un optimizador del tipo AdamW, que solo supone un cambio en la tasa de descarte de pesos, solo perceptible en entrenamientos largos con los valores por defecto; como medida de pérdida usamos `BinaryCrossentropy`, siendo lo estandar para el tipo de clasificación que vamos a usar; la métrica principal que usaremos es la `BinaryAccuracy` para saber la exactitud del modelo.

En cuanto al RandomForest, la creación que usamos es completamente por defecto, siendo los campos que asignamos `n_estimators=100` y la semilla de creación a 42.

La combinación final de hiperparámetros elegida en base a los resultados anteriores es la siguiente:

```python
final_params = {
    "columntransformer__tfidfvectorizer__min_df": 2,
    "logisticregression__C": 1,
    "logisticregression__class_weight": None,
}
```

### 5.3. Entrenamiento

Con ambos modelos usaremos la partición mencionada anteriormente de 80/20 entre entrenamiento y testing.

Para el entrenamiento establecemos un `callback` sencillo que pare el entrenamiento cuando se pierda exactitud durante 3 épocas seguidas, restaurando el valor. Debido al pequeño tamaño del dataset, establecemos un tamaño de lotes de entrenamiento de 8, y establecemos los datos de X_train e y_train como los datos de validación. Importante guardar en una variable las métricas del entrenamiento.

El entrenamiento del RandomForest será solo pasando los datos de entrenamiento a la función, en este caso no se devolverá una variable con métricas.

### 5.4. Evaluación

La evaluación que usamos para ambos modelos será la exactitud del modelo y la pérdida, ya que disponemos de un volumen de datos bastante bajo, solo nos intentaremos alejar lo máximo posible del azar sin que haya un sobreajuste a los datos. Con esto el rango que conseguimos para la exactitud ronda el 70%.

### 5.5. Predicción


### 6. Flujo Airflow

El flujo de trabajo se gestiona a través de un DAG (Triaje_DAG) dividido en varias tareas modulares que se ejecutan en colas específicas de Celery:

|                              | Descripción                                                                                                                       | Queue              | Fichero              |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ------------------ | -------------------- |
| **leer_dataset**             | Carga inicial del archivo *triaje.csv.gz* y guarda los datos en un dataframe.                                                     | ``low_tier_tasks`` | *data_processing.py* |
| **procesar_dataset**         | Limpia los datos y los filtra para guardarlos en un archivo csv (triaje_limpio.csv). Elimina valores nulos y binariza ``acuity``. | ``cpu_tasks``      | *data_processing.py* |
| **procesar_dataset_edstays** | Procesa los datos de *triaje.csv.gz* y *edstays.csv.gz* (triage_ed_limpio.csv).                                                   | ``cpu_tasks``      | *data_processing.py* |
| **entrenar_modelo_rf**       | Entrena en paralelo un clasificador Random Forest Classifier. Evalúa precisión y registra modelo y matriz de confusión.           | ``gpu_tasks``      | *model_training.py*  |
| **entrenar_modelo_tf**       | Entrena en paralelo una red neuronal en Keras y guarda artefactos MLflow.                                                         | ``gpu_tasks``      | *model_training.py*  |
| **entrenar_modelo_lr**       | Entrena en paralelo un clasificador Logistic Regression                                                                           | ``gpu_tasks``      | *model_training.py*  |

**Estructura de ejecución del flujo:**

![alt text](img/image.png)

**Gestión de colas (Celery):**
- ``low_tier_tasks`` utilizado para tareas ligeras como lectura e inspección básica de datos, ya que tiene una memoria limitada de 512m.
- ``cpu_tasks`` para el procesamiento intensivo de datos y transformación con Pandas y NumPy.
- ``gpu_tasks`` para Workers dedicados al entrenamiento de modelos de Machine Learning.


### 7. Versionado con MLflow

Los artefactos se gestionan a través del archivo *src/triaje/utils.py* mediante las funciones ``plot_keras_history`` y ``plot_matriz_confusion`` que se registran directamente en la interfaz MLflow. También se añade la gráfica de comparación de las diferentes convinaciones de parámetros del modelo para escoger cuál es mejor para este caso con la función ``plot_grid_search_results``.

**Matriz de confusión de la Red Neuronal Keras:**

![alt text](img/cm_Keras_Red_Neuronal.png)

**Curvas de Loss y Acuracy de entrenamiento de Keras:**

![alt text](img/keras_history.png)

**Gráfica comparación Loss y Accuracy de diferentes combinaciones de hiperparámetros:**



**Métricas concretas resultantes del modelo ``Clasificador_TF_Triaje`` con Keras:**

![alt text](img/image-2.png)

**Métricas concretas resultantes del modelo ``Clasificador_RF_Triaje`` con Random Forest:**

![alt text](img/image-3.png)

**Métricas concretas resultantes del modelo ``Clasificador_RF_Triaje`` con Random Forest:**





Los **modelos** se quedan registrados automáticamente al lanzar el DAG en el **Model Registry** de MLflow con los nombres ``Clasificador_TF_Triaje`` y ``Clasificador_RF_Triaje`` para el control de versiones.

![alt text](img/image-1.png)

### 8. Limitaciones y consideraciones

Limitaciones:
- Dataset de Triaje con pocos registros útiles (190) para entrenamiento del modelo.

