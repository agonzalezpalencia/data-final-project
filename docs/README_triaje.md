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



## 5. El modelo

### 5.1. Pruebas

### 5.2. Creación

### 5.3. Entrenamiento

### 5.4. Evaluación

### 5.5. Predicción


### 6. Flujo Airflow


### 7. Versionado con MLflow


### 8. Limitaciones y consideraciones

