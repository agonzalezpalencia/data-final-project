from datetime import datetime, timedelta
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.models import taskinstance as ti
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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


################################### - Método entrenamiento 3er DAG - ########################################
def entrenar_modelos():
    df = pd.read_csv("/opt/airflow/plugins/triage_limpio.csv")

    X = df[['temperature', 'heartrate','resprate', 'o2sat', 'sbp', 'dbp','pain']].astype({
        'temperature': 'float32',
        'heartrate': 'float32',
        'resprate': 'float32',
        'o2sat': 'float32',
        'sbp': 'float32',
        'dbp': 'float32',
        'pain': 'float32'
    }).to_numpy()

    y = df[['acuity']].to_numpy(dtype=np.float32)

    y = y.ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # Modelo de Scikit-Learn de tipo de Bosque Aleatorio
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Modelo de Tensorflow Keras de clasificación binaria
    normalizator = tf.keras.layers.Normalization(axis=-1)
    tf.keras.utils.set_random_seed(42)

    normalizator.adapt(X_train)

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(7,)),
        normalizator,
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_binary_accuracy',
        patience=3,
        restore_best_weights=True
    )

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=0.01
    )

    loss = tf.keras.losses.BinaryCrossentropy()

    metric = tf.keras.metrics.BinaryAccuracy()

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[metric]
    )

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment('Experimento Entreno')

    with mlflow.start_run(run_name="RandomForest_Base") as run_rf:
        mlflow.sklearn.autolog()
        rf.fit(X_train, y_train)
        accuracy_rf = rf.score(X_test, y_test)      
        id_rf = run_rf.info.run_id
        print("Accuracy RandomForest:", accuracy_rf)

    with mlflow.start_run(run_name="Keras_Binary") as run_tf:
        mlflow.tensorflow.autolog()
        history = model.fit(X_train, y_train, epochs=25, batch_size=8, callbacks=[early_stopping], validation_data=(X_test, y_test))
        accuracy_tf = history.history['val_binary_accuracy'][-1]
        id_tf = run_tf.info.run_id
        print("Accuracy Tensorflow:", accuracy_tf)

    if (accuracy_tf > accuracy_rf):
        mejor_run_id = id_tf
    else:
        mejor_run_id = id_rf

    model_uri = f"runs:/{mejor_run_id}/model"
    mlflow.register_model(model_uri=model_uri, name="Clasificador_Triaje")
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

    leer_dataset_task = PythonOperator(
        task_id='leer_dataset',
        python_callable=leer_dataset,
        queue="low_tier_tasks"
    )

    disparar_procesado_datos = TriggerDagRunOperator(
        task_id="disparar_dag_procesado",
        trigger_dag_id="data_processing",
        wait_for_completion=False
    )

    leer_dataset_task >> disparar_procesado_datos
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

    procesar_dataset_task = PythonOperator(
        task_id='procesar_dataset',
        python_callable=procesar_dataset,
        queue="cpu_tasks"
    )

    disparar_entrenamiento_modelos = TriggerDagRunOperator(
        task_id="disparar_dag_entrenamiento",
        trigger_dag_id="model_training",
        wait_for_completion=False
    )

    procesar_dataset_task >> disparar_entrenamiento_modelos
#############################################################################################################


########################### - DAG Modelos_1 - Utiliza MLFlow para entrenar - ################################
with DAG(
    'model_training',
    default_args=default_args,
    description='Un simple DAG de prueba Hello World',
    schedule_interval=None,
    catchup=False,
    tags=['example'],
) as dag:

    entrenar_modelos_task = PythonOperator(
        task_id='entrenar_modelos',
        python_callable=entrenar_modelos,
        queue='gpu_tasks'
    )

    entrenar_modelos_task