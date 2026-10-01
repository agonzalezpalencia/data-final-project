from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
import pandas as pd
import numpy as np

################################## - Método lectura de 1er DAG - ############################################
def leer_dataset():
    triaje = pd.read_csv("/opt/airflow/datos/mimic-iv-ed-demo-2.2/mimic-iv-ed-demo-2.2/ed/triage.csv.gz")
    print(f"Filas: {triaje.shape[0]}")
    print(f"Columnas: {triaje.shape[1]}")

    return "Finished reading!"
#############################################################################################################


################################### - Método procesado de 2do DAG - #########################################
def procesar_dataset():
    triaje = pd.read_csv("/opt/airflow/datos/mimic-iv-ed-demo-2.2/mimic-iv-ed-demo-2.2/ed/triage.csv.gz")
    # Eliminamos las filas con los valores nulos en los signos vitales necesarios para entrenamiento
    df = triaje[triaje['acuity'].notna() & triaje['o2sat'].notna() & triaje['temperature'].notna() & triaje['heartrate'].notna()].drop(columns=['subject_id', 'stay_id', 'chiefcomplaint'])
    
    # Sustituimos los valores que no se pueden registrar con un 0 (inconscientes o no pueden hablar)
    df['pain'] = df['pain'].replace({'UA':0, 'unable':0, 'uta':0, 'ett':0, 'o':0})

    #
    df['pain'] = pd.to_numeric(df['pain'])

    # Binarización de 'acuity' con valores de 0 y 1, 0 siendo urgente y 1 siendo NO urgente
    df['acuity'] = np.round(df['acuity'] / 5)

    print(df[:40])

    # Guardamos el resultado del triaje limpio
    ruta_csv_limpio = "/opt/airflow/plugins/triage_limpio.csv"
    df.to_csv(ruta_csv_limpio)

    return ruta_csv_limpio
#############################################################################################################


################################ - Argumentos comunes - #####################################################
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_since': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}
#############################################################################################################

################################ - DAG Lectura - Comprueba archivo - ########################################
with DAG(
    'data_reading',
    default_args=default_args,
    description='Un simple DAG de prueba Hello World',
    schedule_interval='@daily',
    start_date=datetime(2026, 10, 1),
    catchup=False,
    tags=['example'],
) as dag:

    read_dataset_task = PythonOperator(
        task_id='leer_dataset',
        python_callable=leer_dataset,
        queue="low_tier_tasks"
    )

    disparar_procesado_datos = TriggerDagRunOperator(
        task_id="disparar_dag_procesado",
        trigger_dag_id="data_processing",
        wait_for_completion=False
    )

    read_dataset_task >> disparar_procesado_datos
#############################################################################################################


################################# - DAG Procesado - Prepara CSV Entreno - ###################################
with DAG(
    'data_processing',
    default_args=default_args,
    description='Un simple DAG de prueba Hello World',
    schedule_interval=None,
    catchup=False,
    tags=['example'],
) as dag:

    process_dataset_task = PythonOperator(
        task_id='procesar_dataset',
        python_callable=procesar_dataset,
        queue="cpu_tasks"
    )

    process_dataset_task
#############################################################################################################