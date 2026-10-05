from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator
from airflow import DAG
from src.sepsis.data_processing import (sepsis_ds_charge, 
sepsis_clinic_columns_fill, 
sepsis_feature_engineering, 
sepsis_medians_fill,
sepsis_null_columns_drop,
sepsis_null_constants_drop,
sepsis_outliers_filtering,
sepsis_train_test_split,
sepsis_train_val_test_split)
from src.sepsis.model_training import sepsis_rd_train

################################ - Argumentos comunes - #####################################################
default_args = {
    'owner': 'agonzalezpalencia',
    'depends_on_past': False,
    'email_since': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}
#############################################################################################################

############################################ - DAG Sepsis - #################################################
with DAG(
    'sepsis_dag',
    default_args=default_args,
    description='DAG que procesa los pacientes de Sepsis al completo y entrena un modelo de Scikit Learn usando el algoritmo RandomForest',
    schedule_interval='@monthly',
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
#############################################################################################################
