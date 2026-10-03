import pandas as pd
import numpy as np

def leer_dataset(ti):
    csv_base = "/opt/airflow/datos/mimic-iv-ed-demo-2.2/mimic-iv-ed-demo-2.2/ed/triage.csv.gz"
    triaje = pd.read_csv(csv_base)
    print(f"Filas: {triaje.shape[0]}")
    print(f"Columnas: {triaje.shape[1]}")

    return csv_base

def procesar_dataset(ti):
    csv_base = ti.xcom_pull(task_ids="leer_dataset")
    triaje = pd.read_csv(csv_base)
    
    df = triaje[triaje['acuity'].notna() & triaje['o2sat'].notna() & triaje['temperature'].notna() & triaje['heartrate'].notna()].drop(columns=['subject_id', 'stay_id', 'chiefcomplaint'])
    
    df['pain'] = df['pain'].replace({'UA':0, 'unable':0, 'uta':0, 'ett':0, 'o':0})
    df['pain'] = pd.to_numeric(df['pain'])
    df['acuity'] = np.round(df['acuity'] / 5)

    print(df[:40])

    ruta_csv_limpio = "/opt/airflow/plugins/triage_limpio.csv"
    df.to_csv(ruta_csv_limpio, index=False) # Recomendado: index=False para evitar la columna Unnamed: 0

    return ruta_csv_limpio