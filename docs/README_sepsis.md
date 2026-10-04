# Proyecto Hospital - Sepsis

## 1. Contexto y objetivos

El contexto de este dataset es la predicción del estado de sepsis de pacientes que se encuentran actualmente en observación en urgencias. Las muestras son temporales, es decir, un paciente tendrá X número de registros, referenciando las diferentes observaciones de sus signos vitales a lo largo de su estancia.
El campo más importante será el que confirma si durante X observación el paciente está sufriendo de sepsis o no.
El objetivo que nos hemos propuesto es la creación de un modelo que puede discernir según los signos vitales de un paciente en una observación pueda determinar si sufre de sepsis o no.

## 2. Dataset (Sepsis - PhysioNet)

Dataset: [Sepsis](https://physionet.org/content/challenge-2019/1.0.0/)

Número de registros brutos: 1552210

| # | Campo | Descripción |
|---:|---|---|
| 1 | HR | Heart rate (beats per minute) |
| 2 | O2Sat | Pulse oximetry (%) |
| 3 | Temp | Temperature (Deg C) |
| 4 | SBP | Systolic BP (mm Hg) |
| 5 | MAP | Mean arterial pressure (mm Hg) |
| 6 | DBP | Diastolic BP (mm Hg) |
| 7 | Resp | Respiration rate (breaths per minute) |
| 8 | EtCO2 | End tidal carbon dioxide (mm Hg) |
| 9 | BaseExcess | Measure of excess bicarbonate (mmol/L) |
| 10 | HCO3 | Bicarbonate (mmol/L) |
| 11 | FiO2 | Fraction of inspired oxygen (%) |
| 12 | pH | N/A |
| 13 | PaCO2 | Partial pressure of carbon dioxide from arterial blood (mm Hg) |
| 14 | SaO2 | Oxygen saturation from arterial blood (%) |
| 15 | AST | Aspartate transaminase (IU/L) |
| 16 | BUN | Blood urea nitrogen (mg/dL) |
| 17 | Alkalinephos | Alkaline phosphatase (IU/L) |
| 18 | Calcium | (mg/dL) |
| 19 | Chloride | (mmol/L) |
| 20 | Creatinine | (mg/dL) |
| 21 | Bilirubin_direct | Bilirubin direct (mg/dL) |
| 22 | Glucose | Serum glucose (mg/dL) |
| 23 | Lactate | Lactic acid (mg/dL) |
| 24 | Magnesium | (mmol/dL) |
| 25 | Phosphate | (mg/dL) |
| 26 | Potassium | (mmol/L) |
| 27 | Bilirubin_total | Total bilirubin (mg/dL) |
| 28 | TroponinI | Troponin I (ng/mL) |
| 29 | Hct | Hematocrit (%) |
| 30 | Hgb | Hemoglobin (g/dL) |
| 31 | PTT | Partial thromboplastin time (seconds) |
| 32 | WBC | Leukocyte count (count*10^3/µL) |
| 33 | Fibrinogen | (mg/dL) |
| 34 | Platelets | (count*10^3/µL) |
| 35 | Age | Years (100 for patients 90 or above) |
| 36 | Gender | Female (0) or Male (1) |
| 37 | Unit1 | Administrative identifier for ICU unit (MICU) |
| 38 | Unit2 | Administrative identifier for ICU unit (SICU) |
| 39 | HospAdmTime | Hours between hospital admit and ICU admit |
| 40 | ICULOS | ICU length-of-stay (hours since ICU admit) |
| 41 | SepsisLabel | Whether the patient has sepsis or not |
| 42 | patient_id | Patient identifier |
| 43 | hospital | Hospital A or B |

### 2.1. Variable objetivo

El objetivo de nuestro modelo será `SepsisLabel` mediante predicciones con el uso del resto de las variables.

## 3. Análisis del dataset

Según la documentación oficial de este dataset, los primeros 8 campos serán los signos vitales comunes del paciente, los siguientes hasta `SepsisLabel` son valores de laboratorio, dadas pruebas u otras mediciones.

El reparto de registros dada la variable objetivo es un ~2%, siendo el 98% registros donde los pacientes no sufren de sepsis, lo que no añade un reto de sobreajuste para el modelo a desarrollar. Entre las soluciones que barajamos en un inicio fue el recoger de forma aleatoria una sección de los registros sin sepsis hasta conseguir un número equilibrado 50-50 de ambos tipos, pero al ser un reparto tan desproporcionado fue descartado.

Otro aspecto notable del dataset es el hecho de que no en todas las instancias de medición habrá datos de laboratorio, sino que se realizan en periodos de tiempo mayores a los registrados, lo que crea valores nulos en los campos de laboratorio.

### 3.1. Matriz correlación


## 4. Limpieza del dataset

### 4.1. Descartes y transformaciones

> Columnas eliminadas, limpiadas y añadidas.

### 4.2. Resultado limpieza

Número de registros tras limpieza: 
Campos finales del dataset: 
    1. dddd

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

