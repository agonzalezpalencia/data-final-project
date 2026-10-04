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
    os.makedirs(tmp_path, exist_ok=True)
    ruta = tmp_path + nombre + ".parquet"
    df.to_parquet(ruta)
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

def sepsis_clinic_columns_fill(ti):

    # Recuperamos el dataset de sepsis desde la tarea anterior
    sepsis_df = pd.read_parquet(ti.xcom_pull(task_ids="filtrado_outliers"))

    clinic_columns = [c for c in sepsis_df.columns if c not in no_clinic_columns]

    sepsis_df[clinic_columns] = sepsis_df.groupby('patient_id')[clinic_columns].ffill()
    
    return guardar(sepsis_df, "03_ffill")


def sepsis_null_columns_drop(ti):

    # Recuperamos el dataset de sepsis desde la tarea anterior
    sepsis_df = pd.read_parquet(ti.xcom_pull(task_ids="rellenar_columnas_datos_vitales"))

    null_values = sepsis_df.isna().mean().sort_values(ascending=False)

    sepsis_df = sepsis_df.drop(columns=null_values[null_values > 0.80].index)
    
    return guardar(sepsis_df, "04_columnas_nulas")


def sepsis_null_constants_drop(ti):
    # Recuperamos el dataset de sepsis desde la tarea anterior
    sepsis_df = pd.read_parquet(ti.xcom_pull(task_ids="eliminar_columnas_valores_nulos"))

    clinic_columns = [c for c in sepsis_df.columns if c not in no_clinic_columns]
    sepsis_df = sepsis_df.dropna(subset=clinic_columns, how="all")
    
    return guardar(sepsis_df, "05_constantes_nulas")


def sepsis_feature_engineering(ti):

    # Recuperamos el dataset de sepsis desde la tarea anterior
    sepsis_df = pd.read_parquet(ti.xcom_pull(task_ids="eliminar_constantes_nulas"))

    # Variables por registro o observaciñon que no tienen datos rellenados
    clinic_columns = [c for c in sepsis_df.columns if c not in no_clinic_columns]
    sepsis_df['reg_w_data'] = sepsis_df[clinic_columns].isna().sum(axis=1)

    # Índice de shock anafiláctico, nos sirve para comparar la gravedad del paciente
    sepsis_df['shock_index'] = sepsis_df['HR'] / sepsis_df['SBP']
    
    return guardar(sepsis_df, "06_features")

def sepsis_train_test_split(ti):

    # Recuperamos el dataset de sepsis desde la tarea anterior
    sepsis_df = pd.read_parquet(ti.xcom_pull(task_ids="feature_engineering"))

    patients = sepsis_df['patient_id'].drop_duplicates().sample(frac=1, random_state=42)
    test_ids = patients[:int(len(patients) * 0.2)]

    pre_test = sepsis_df['patient_id'].isin(test_ids)
    train = sepsis_df[~pre_test].copy()
    test = sepsis_df[pre_test].copy()

    print(len(train), len(test))
    return guardar(train, "07_train"), guardar(test, "07_test")

def sepsis_medians_fill(ti):

    # Recuperamos el dataset de train y de test desde la tarea anterior
    ruta_train, ruta_test = ti.xcom_pull(task_ids="split_train_test")
    train = pd.read_parquet(ruta_train)
    test = pd.read_parquet(ruta_test)

    medians = train.median(numeric_only=True)
    train = train.fillna(medians)
    test = test.fillna(medians)

    # Verificación de que no ha quedado ningún dato vacío y todos estan rellenados
    print(train.isna().sum().sum(), test.isna().sum().sum())

    return guardar(train, "08_train"), guardar(test, "08_test")

def sepsis_train_val_test_split(ti):

    ruta_train, ruta_test = ti.xcom_pull(task_ids="rellenar_datos_medianas")
    train = pd.read_parquet(ruta_train)
    test = pd.read_parquet(ruta_test)

    ID = ["patient_id", "hospital"]

    # Aplicamos la misma lógica que hemos utilizado para separar train y test, pero esta vez en train para obtneer los datos de validación
    train_patients = train["patient_id"].drop_duplicates().sample(frac=1, random_state=42)
    val_ids = train_patients[:int(len(train_patients) * 0.2)]
    in_val = train["patient_id"].isin(val_ids).to_numpy()

    tr = train[~in_val].drop(columns=ID).astype("float32")
    val = train[~in_val].drop(columns=ID).astype("float32")
    test = test.drop(columns=ID).astype("float32")

    return guardar(tr, "09_tr"), guardar(val, "09_val"), guardar(test, "09_test")

def sepsis_rd_train(ti):

    ruta_tr, ruta_val, ruta_test = ti.xcom_pull(task_ids="separar_train_validation_test")
    tr = pd.read_parquet(ruta_tr)
    val = pd.read_parquet(ruta_val)
    test = pd.read_parquet(ruta_test)

    X_tr = tr.drop(columns=['SepsisLabel']).to_numpy()
    y_tr = tr['SepsisLabel'].to_numpy()
    X_test = val.drop(columns=['SepsisLabel']).to_numpy()
    y_test = val['SepsisLabel'].to_numpy()
    X_val = test.drop(columns=['SepsisLabel']).to_numpy()
    y_val = test['SepsisLabel'].to_numpy()

    rf = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=50,
        max_samples=0.2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment('Experimento Entreno')

    with mlflow.start_run(run_name="RandomForest_Base") as run_rf:
        
        rf.fit(X_tr, y_tr)
        mlflow.sklearn.log_model(rf, "model")

        pred_val = rf.predict_proba(X_val)[:, 1]
        pred_rf = rf.predict_proba(X_test)[:, 1]

        print("AUPRC validación:", average_precision_score(y_val, pred_val))
        print("AUPRC test:", average_precision_score(y_test, pred_rf))
        print("Pérdida validación:", log_loss(y_val, pred_val))
        print("Pérdida test:", log_loss(y_test, pred_rf))
        print("Accuracy validación:", accuracy_score(y_val, pred_val > 0.5))
        print("Accuracy test:", accuracy_score(y_test, pred_rf > 0.5))

        id_rf = run_rf.info.run_id

    model_uri = f"runs:/{id_rf}/model"
    mlflow.register_model(model_uri=model_uri, name="Clasificador_RF_Sepsis")

################################ - Argumentos comunes - #####################################################
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_since': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}
#############################################################################################################

################################ - DAG Sepsis - ########################################
with DAG(
    'sepsis_dag',
    default_args=default_args,
    description='DAG que procesa los pacientes de Sepsis al completo y entrena un modelo de Scikit Learn usando el algoritmo RandomForest',
    schedule_interval=None,
    start_date=datetime(2026, 10, 1),
    catchup=False,
    max_active_runs=1,
    tags=['sepsis', 'agonzalezpalencia'],
) as dag:

    sepsis_ds_charge_task = PythonOperator(
        task_id='cargar_dataset',
        python_callable=sepsis_ds_charge,
        queue="cpu_tasks"
    )

    sepsis_outliers_filtering_task = PythonOperator(
        task_id='filtrado_outliers',
        python_callable=sepsis_outliers_filtering,
        queue="cpu_tasks"
    )

    sepsis_clinic_columns_fill_task = PythonOperator(
        task_id='rellenar_columnas_datos_vitales',
        python_callable=sepsis_clinic_columns_fill,
        queue="cpu_tasks"
    )

    sepsis_null_columns_drop_task = PythonOperator(
        task_id='eliminar_columnas_valores_nulos',
        python_callable=sepsis_null_columns_drop,
        queue="cpu_tasks"
    )

    sepsis_null_constants_drop_task = PythonOperator(
        task_id='eliminar_constantes_nulas',
        python_callable=sepsis_null_constants_drop,
        queue="cpu_tasks"
    )

    sepsis_feature_engineering_task = PythonOperator(
        task_id='feature_engineering',
        python_callable=sepsis_feature_engineering,
        queue="cpu_tasks"
    )

    sepsis_train_test_split_task = PythonOperator(
        task_id='split_train_test',
        python_callable=sepsis_train_test_split,
        queue="cpu_tasks"
    )

    sepsis_medians_fill_task = PythonOperator(
        task_id='rellenar_datos_medianas',
        python_callable=sepsis_medians_fill,
        queue="cpu_tasks"
    )

    sepsis_train_val_test_split_task = PythonOperator(
        task_id='separar_train_validation_test',
        python_callable=sepsis_train_val_test_split,
        queue="cpu_tasks"
    )

    sepsis_rd_train_task = PythonOperator(
        task_id='entrenar_modelo',
        python_callable=sepsis_rd_train,
        queue="gpu_tasks"
    )

(
    sepsis_ds_charge_task >> 
    sepsis_outliers_filtering_task >> 
    sepsis_clinic_columns_fill_task >> 
    sepsis_null_columns_drop_task >> 
    sepsis_null_constants_drop_task >> 
    sepsis_feature_engineering_task >> 
    sepsis_train_test_split_task >> 
    sepsis_medians_fill_task >> 
    sepsis_train_val_test_split_task >> 
    sepsis_rd_train_task
)

