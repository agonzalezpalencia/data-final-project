from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

# Importaciones locales desde la carpeta src/triaje
from src.triaje.data_processing import leer_dataset, procesar_dataset, procesar_datset_edstays
from src.triaje.model_training import entrenar_modelo_tf, entrenar_modelo_rf, entrenar_modelo_lr

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_since': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'Triaje_DAG',
    default_args=default_args,
    description='DAG que procesa el Dataset de Triaje y entrena modelos en paralelo (RF y Keras)',
    schedule_interval='@monthly',
    start_date=datetime(2026, 10, 4),
    catchup=False,
    tags=['urgencias', 'mlflow'],
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
    
    procesar_dataset_ed_task = PythonOperator(
        task_id='procesar_dataset_edstays',
        python_callable=procesar_datset_edstays,
        queue="cpu_tasks"
    )

    entrenar_modelo_tf_task = PythonOperator(
        task_id='entrenar_modelo_tf',
        python_callable=entrenar_modelo_tf,
        queue='gpu_tasks'
    )
    
    entrenar_modelo_rf_task = PythonOperator(
        task_id='entrenar_modelo_rf',
        python_callable=entrenar_modelo_rf,
        queue='gpu_tasks'
    )

    entrenar_modelo_lr_task = PythonOperator(
        task_id='entrenar_modelo_lr',
        python_callable=entrenar_modelo_lr,
        queue='gpu_tasks'
    )

    # Flujo: Lineal al principio, paralelo al final
    leer_dataset_task >> [procesar_dataset_task, procesar_dataset_ed_task] 
    
    procesar_dataset_task >> [entrenar_modelo_tf_task, entrenar_modelo_rf_task]
    
    procesar_dataset_ed_task >> entrenar_modelo_lr_task
