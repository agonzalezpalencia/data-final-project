from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

# Importaciones locales desde la carpeta src/triaje
from src.triaje_version_m.data_processing_m import leer_dataset, procesar_dataset
from src.triaje_version_m.model_training_m import entrenar_modelo_rf_m

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_since': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'Triaje_DAG_m',
    default_args=default_args,
    description='DAG que procesa el Dataset de Triaje y entrena modelo Random Forest',
    schedule_interval='@daily',
    start_date=datetime(2026, 10, 3),
    catchup=False,
    tags=['urgencias', 'mlflow', 'version_m', 'random_forest'],
) as dag:

    leer_dataset_task = PythonOperator(
        task_id='leer_dataset',
        python_callable=leer_dataset,
        queue="low_tier_tasks"
    )

    procesar_dataset_task = PythonOperator(
        task_id='procesar_dataset',
        python_callable=procesar_dataset,
        queue="cpu_tasks"
    )
    
    entrenar_modelo_rf_m_task = PythonOperator(
        task_id='entrenar_modelo_rf_m',
        python_callable=entrenar_modelo_rf_m,
        queue='gpu_tasks'
    )

    # Flujo: Lineal al principio, paralelo al final
    leer_dataset_task >> procesar_dataset_task >> [entrenar_modelo_rf_m_task]