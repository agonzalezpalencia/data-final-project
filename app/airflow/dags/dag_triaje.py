from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.models.param import Param

# Importaciones locales desde la carpeta src/triaje
from src.triaje.data_processing import leer_dataset, procesar_dataset, procesar_datset_edstays
from src.triaje.model_training import entrenar_modelo_tf, entrenar_modelo_rf, entrenar_modelo_lr
from src.triaje.utils import enrutador_procesado, enrutador_modelos

from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator

import src.triaje.data_processing
import src.triaje.model_training
import src.triaje.utils

default_args = {
    'owner': 'MLOps',
    'depends_on_past': False,
    'email_since': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

# función para Branch para decidir qué modelo reentrenar en función del umbral de accuracy obtenido en la primera ejecución
def evaluar_modelos(ti):
    umbral = 0.80
    # Obtener métricas de accuracy del entrenamiento de cada modelo desde XCom
    modelos = {
        "entrenar_modelo_tf": "reentrenar_modelo_tf",
        "entrenar_modelo_rf": "reentrenar_modelo_rf",
        "entrenar_modelo_lr": "reentrenar_modelo_lr",
    }
    dag_run = ti.get_dagrun()
    para_reentrenar = []
    # cada modelo por debajo del umbral se reentrena
    for train_id, retrain_id in modelos.items():
        # Si la task no se ejecutó con éxito (skipped por el router), se ignora
        if dag_run.get_task_instance(train_id).state != "success":
            continue
        acc = ti.xcom_pull(task_ids=train_id)
        if acc is None or acc < umbral:
            para_reentrenar.append(retrain_id)
    return para_reentrenar or "sin_reentreno"

with DAG(
    'Triaje_DAG',
    default_args=default_args,
    description='DAG que procesa el Dataset de Triaje y entrena modelos en paralelo (RF, LR y Keras)',
    schedule_interval='@monthly',
    start_date=datetime(2026, 10, 4),
    catchup=False,
    max_active_runs=1,
    tags=['urgencias', 'mlflow'],
    params={
        # Poner: 'TF' para Tensorflow, 'RF' para RandomForestClassifier y 'LR' para LogisticRegression
        "modelos_a_entrenar":"TF RF LR"
    },
) as dag:

    leer_dataset_task = PythonOperator(
        task_id='leer_dataset',
        python_callable=leer_dataset,
        queue="low_tier_tasks"
    )

    enrutador_procesado_task = BranchPythonOperator(
        task_id='enrutador_procesado',
        python_callable=enrutador_procesado,
        queue='low_tier_tasks'
    )

    procesar_dataset_task = PythonOperator(
        task_id='procesar_dataset',
        python_callable=procesar_dataset,
        queue="cpu_tasks"
    )

    # Si está seleccionado en los parámetros LR se ejecutará esta rama
    procesar_dataset_ed_task = PythonOperator(
        task_id='procesar_dataset_edstays',
        python_callable=procesar_datset_edstays,
        queue="cpu_tasks"
    )

    enrutador_modelos_task = BranchPythonOperator(
        task_id='enrutador_modelos',
        python_callable=enrutador_modelos,
        queue='low_tier_tasks'
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
    
    evaluar_modelos_task = BranchPythonOperator(
        task_id='evaluar_modelos',
        python_callable=evaluar_modelos,
        queue='low_tier_tasks',
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS
    )
    
    reentrenar_modelo_tf_task = PythonOperator(
        task_id='reentrenar_modelo_tf',
        python_callable=entrenar_modelo_tf,
        # pasar hiperparámetros para reentrenar el modelo
        op_kwargs={"epochs": 60, "neur": 64, "learning_rate": 0.003, "run_name": "Keras_Reentreno"},
        queue='gpu_tasks'
    )
    
    reentrenar_modelo_rf_task = PythonOperator(
        task_id='reentrenar_modelo_rf',
        python_callable=entrenar_modelo_rf,
        # pasar hiperparámetros para reentrenar el modelo
        op_kwargs={"n_estimators": 500, "run_name": "RandomForest_Reentreno"},
        queue='gpu_tasks'
    )
    
    reentrenar_modelo_lr_task = PythonOperator(
        task_id='reentrenar_modelo_lr',
        python_callable=entrenar_modelo_lr,
        # pasar hiperparámetros para reentrenar el modelo
        op_kwargs={"c_values": (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100), "run_name": "LogisticRegression_Reentreno"},
        queue='gpu_tasks'
    )
    
    sin_reentreno_task = EmptyOperator(task_id='sin_reentreno')
    

    abandonar_flujo_task = EmptyOperator(
        task_id="abandonar_flujo"
    )

    # Flujo: Lineal al principio, paralelo al final
    leer_dataset_task >> enrutador_procesado_task

    enrutador_procesado_task >> [procesar_dataset_task, procesar_dataset_ed_task, abandonar_flujo_task]

    procesar_dataset_task >> enrutador_modelos_task
    enrutador_modelos_task >> [entrenar_modelo_tf_task, entrenar_modelo_rf_task]

    procesar_dataset_ed_task >> entrenar_modelo_lr_task
    
    [entrenar_modelo_tf_task, entrenar_modelo_rf_task, entrenar_modelo_lr_task] >> evaluar_modelos_task 
    
    evaluar_modelos_task >> [reentrenar_modelo_tf_task, reentrenar_modelo_rf_task, reentrenar_modelo_lr_task, sin_reentreno_task]
