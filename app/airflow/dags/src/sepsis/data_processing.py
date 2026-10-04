from datetime import datetime, timedelta
import mlflow
import mlflow.sklearn
from airflow import DAG
from airflow.operators.python import PythonOperator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, accuracy_score
from sklearn.metrics import log_loss
import pandas as pd
import numpy as np
import os

# Ruta temporal donde se irán almacenando los datasets a medida que los vayamos filtrando 
tmp_path = "/opt/airflow/datos/sepsis_tmp/"

# Método al que pasamos el dataframe y el nombre del archivo para poder almacenarlo en la carpeta temporal
def guardar(df, nombre):
    os.makedirs(tmp_path, exist_ok=True) # Creación del directorio
    ruta = tmp_path + nombre + ".parquet" # Definimos la ruta en base al nombre que hayamos pasado y lo guardamos con la extensiñon de '.parquet'
    df.to_parquet(ruta) # Convertimos el Dataframe a .parquet y lo guardamos en la ruta definida 
    return ruta

# Columnas utilizadas en varios procesos de filtrado, para generalizar y no declararla varias veces, la declaramos como variable una única vez y la vamos reutilizando
no_clinic_columns = [
    "patient_id", "hospital", "Age", "Gender", "Unit1", "Unit2",
    "HospAdmTime", "ICULOS", "SepsisLabel"
]



# Carga inicial de todas las entradas que contienen tanto la carpeta training_setA como training_setB de Sepsis
def sepsis_ds_charge(ti):

    # Leemos el parquet que hemos procesado previamente desde el cuaderno de Jupyter (notebook_abel_sepsis.ipynb)
    sepsis_df = pd.read_parquet(
        "/opt/airflow/parquets/sepsis_parquet_raw.parquet"
    )

    return guardar(sepsis_df, "01_carga")



# Filtrado de outliers (Valores atípicos) en base a la primera toma de contacto que se ha realizado con el dataset
def sepsis_outliers_filtering(ti):
    # Definimos las columnas sobre las que aplicaremos las reglas de filtrado de los outliers
    outliers_cols = [
        "HR", "O2Sat", "Temp", "SBP", "MAP", "Resp"
    ]

    # Establecemos los rangos sobre los que vamos a aplicar las reglas de filtrado de los outliers
    outliers_ranges = [
        (20, 250), (50, 100), (30, 43), (40, 280), (20, 200), (4, 70)
    ]

    # Recuperamos el dataset de sepsis desde la tarea anterior
    sepsis_df = pd.read_parquet(ti.xcom_pull(task_ids="cargar_dataset"))

    # Recorremos las columnas
    for i in range(6):
        col = outliers_cols[i]
        lo, hi = outliers_ranges[i]
        sepsis_df.loc[~sepsis_df[col].between(lo, hi),col] = np.nan # Establecemos los valores que se encuentren fuera de los rangos establecidos

    return guardar(sepsis_df, "02_outliers")


# Filtrado de constantes vitales, rellenamos los valores vacíos arrastrando los datos de la hora anterior en caso de ser nulo
def sepsis_clinic_columns_fill(ti):

    # Recuperamos el dataset de sepsis desde la tarea anterior
    sepsis_df = pd.read_parquet(ti.xcom_pull(task_ids="filtrado_outliers"))

    # Obtenemos las columnas que nos aportan constantes vitales del paciente
    clinic_columns = [c for c in sepsis_df.columns if c not in no_clinic_columns]

    # Como tenemos cada paciente ordenada por el ICULOS, podemos rellenar los valores vacíos de esas constantes vitales utilizando ffill(), que nos permite arrastrar el dato 
    # anterior justo un valor por encima, como hemos agrupado por 'patiernt_id' no debemos preocuparnos por solapamiento entre pacientes
    sepsis_df[clinic_columns] = sepsis_df.groupby('patient_id')[clinic_columns].ffill()

    # Guardamos el dataframe con los cambios realizados y lo recogeremos en el siguiente paso
    return guardar(sepsis_df, "03_ffill")


# Filtrado de columnas con más del 80% de sus valores nulos
def sepsis_null_columns_drop(ti):

    # Recuperamos el dataset de sepsis desde la tarea anterior
    sepsis_df = pd.read_parquet(ti.xcom_pull(task_ids="rellenar_columnas_datos_vitales"))

    # Marcamos dentro de las columnas los valores que sean nulos y los que no lo sean    
    null_values = sepsis_df.isna().mean().sort_values(ascending=False)

    # Eliminamos aquellas columnas que tengan mas del 80% de valores nulos, para quitar la mayor cantidad de ruida posible
    sepsis_df = sepsis_df.drop(columns=null_values[null_values > 0.80].index)

    # Guardamos el dataframe con los cambios realizados y lo recogeremos en el siguiente paso
    return guardar(sepsis_df, "04_columnas_nulas")


# Filtrado de datos en las constantes vitales
def sepsis_null_constants_drop(ti):
    # Recuperamos el dataset de sepsis desde la tarea anterior
    sepsis_df = pd.read_parquet(ti.xcom_pull(task_ids="eliminar_columnas_valores_nulos"))

    # Obtenemos las columnas que no coincidan con las clumnas que no tienen datos clínicos, es decir, filtramos las columnas de las constantes vitales
    clinic_columns = [c for c in sepsis_df.columns if c not in no_clinic_columns]
    sepsis_df = sepsis_df.dropna(subset=clinic_columns, how="all") # Eliminamos las filas con horas que contienen todos los valores nulos, ya que solo nos aportan ruido

    # Guardamos el dataframe con los cambios realizados y lo recogeremos en el siguiente paso
    return guardar(sepsis_df, "05_constantes_nulas")

# Feature Engineering sobre las columnas que ya tenemos rellenadas de datos, para sacar más datos concluyentes para el modelo
def sepsis_feature_engineering(ti):

    # Recuperamos el dataset de sepsis desde la tarea anterior
    sepsis_df = pd.read_parquet(ti.xcom_pull(task_ids="eliminar_constantes_nulas"))

    # Variables por registro o observación que no tienen datos rellenados
    clinic_columns = [c for c in sepsis_df.columns if c not in no_clinic_columns]
    sepsis_df['reg_w_data'] = sepsis_df[clinic_columns].isna().sum(axis=1)

    # Índice de shock anafiláctico, nos sirve para comparar la gravedad del paciente
    sepsis_df['shock_index'] = sepsis_df['HR'] / sepsis_df['SBP']

    # Guardamos el dataframe con los cambios realizados y lo recogeremos en el siguiente paso
    return guardar(sepsis_df, "06_features")

# Separación de datos entre Training y Test de cara al entrenamiento del modelo
def sepsis_train_test_split(ti):

    # Recuperamos el dataset de sepsis desde la tarea anterior
    sepsis_df = pd.read_parquet(ti.xcom_pull(task_ids="feature_engineering"))

    # Eliminamos los duplicadois para quedarnos con un ID por paciente y mezclamos los pacientes utilizando la semilla por defecto de 42 
    patients = sepsis_df['patient_id'].drop_duplicates().sample(frac=1, random_state=42)
    test_ids = patients[:int(len(patients) * 0.2)] # Nos quedamos con el 20% de esos registros, estos serán los datos de test

    pre_test = sepsis_df['patient_id'].isin(test_ids) # Marcamos las filas de esos pacientes para poder realizar la correcta separación entre training y test
    train = sepsis_df[~pre_test].copy() # Los que no esten marcados, nos los quedamos para usarlos como datos de entrenamiento
    test = sepsis_df[pre_test].copy() # Los que esten marcados se quedan en test

    print(len(train), len(test))

    # Guardamos los dataframes de train y test para recogerlos mas adelante y aplicar reglas de calidad
    return guardar(train, "07_train"), guardar(test, "07_test")


# Relleno de datos no concluyentes con la mediana de las columnas del dataset
def sepsis_medians_fill(ti):

    # Recuperamos el dataset de train y de test desde la tarea anterior
    path_train, path_test = ti.xcom_pull(task_ids="split_train_test")
    train = pd.read_parquet(path_train)
    test = pd.read_parquet(path_test)

    # Obtenemos las medianas de las columnas y rellenamos los datos nulos con esas mismas medianas
    medians = train.median(numeric_only=True)
    train = train.fillna(medians)
    test = test.fillna(medians)

    # Verificación de que no ha quedado ningún dato vacío y todos estan rellenados
    print(train.isna().sum().sum(), test.isna().sum().sum())

    # Guardamos los dataframes de train y test para recogerlos mas adelante
    return guardar(train, "08_train"), guardar(test, "08_test")

def sepsis_train_val_test_split(ti):

    # Recuperamos el dataset de train y de test desde la tarea anterior
    ruta_train, ruta_test = ti.xcom_pull(task_ids="rellenar_datos_medianas")
    train = pd.read_parquet(ruta_train)
    test = pd.read_parquet(ruta_test)

    ID = ["patient_id", "hospital"]

    # Eliminamos los duplicados para quedarnos con un ID por paciente y mezclamos los pacientes utilizando la semilla por defecto de 42 
    train_patients = train["patient_id"].drop_duplicates().sample(frac=1, random_state=42)
    val_ids = train_patients[:int(len(train_patients) * 0.2)] # El 20% de estos registros se reservan para los datos de validación
    in_val = train["patient_id"].isin(val_ids).to_numpy()

    tr = train[~in_val].drop(columns=ID).astype("float32") # Los que no están marcados, se convierten en nuestros datos para entrenamiento, además de eliminar las columnas con texto
    val = train[in_val].drop(columns=ID).astype("float32") # Los marcados se convierten en nuestros datos de validación , además de eliminar las columnas con texto
    test = test.drop(columns=ID).astype("float32") # Eliminamos las columnas de texto del dataframe

    # Guardamos los dataframes de train, validation y test para recogerlos mas adelante
    return guardar(tr, "09_tr"), guardar(val, "09_val"), guardar(test, "09_test")