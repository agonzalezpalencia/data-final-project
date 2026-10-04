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
    df['pain'] = pd.to_numeric(df['pain'], errors='coerce').fillna(0)
    df['acuity'] = np.round(df['acuity'] / 5)

    print(df[:40])

    ruta_csv_limpio = "/opt/airflow/plugins/triage_limpio.csv"
    df.to_csv(ruta_csv_limpio, index=False) # Recomendado: index=False para evitar la columna Unnamed: 0

    return ruta_csv_limpio

def procesar_datset_edstays(ti):
    ruta_triaje = "/opt/airflow/datos/mimic-iv-ed-demo-2.2/mimic-iv-ed-demo-2.2/ed/triage.csv.gz"
    ruta_edstays = "/opt/airflow/datos/mimic-iv-ed-demo-2.2/mimic-iv-ed-demo-2.2/ed/edstays.csv.gz"
    
    triaje = pd.read_csv(ruta_triaje)
    edstays = pd.read_csv(ruta_edstays, usecols=["stay_id", "intime", "arrival_transport"])
    
    # Hacemos un merge de ambos Dataframes utilizando como clave 'stay_id', manteniendo todos los datos de 'triaje' con left join.
    df = triaje[triaje["acuity"].notna()].merge(edstays, on="stay_id", how="left")
    
    # Creamos la columna 'arrival_hour' a partir del valor intime que contiene la hora exacta de llegada, así no generamos ruido.
    df["arrival_hour"] = pd.to_datetime(df["intime"]).dt.hour 

    # Eliminamos las columnas que no nos proporcionan datos consistentes
    df = df.drop(columns=["subject_id", "stay_id", "intime"])

    # Obtenemos todos los valores de la columna 'pain' y cambiamos el typo que se ha encontrado por un 0.
    pain_txt = df["pain"].astype("string").str.strip().str.lower().replace({"o": "0"})

    # Guardamos la columna 'pain' en una variable pain_num, esta contendrá la columna 'pain' 
    # construida en base a números y completamente normzaliada
    pain_num = df["pain"] = pd.to_numeric(pain_txt, errors="coerce")


    df["pain_not_assessable"] = (pain_txt.notna() & pain_num.isna()).astype(int)
    df["pain"] = pain_num.where(pain_num.between(0, 10))
        
    # Hacemos Feature Engineering sobre los datos que ya tenemos filtrados

    # Creamos un nuevo dato, el índice de shock, que supone la división de los latidos entre la presión arterial sistolica
    df["shock_index"] = df["heartrate"] / df["sbp"]

    # Obtenemos la presión del pulso a partir de la división de la presion arterial sistólica entre la diastólica
    df["pulse_pressure"] = df["sbp"] - df["dbp"]

    # Rellenamos los huecos vacíos con cadenas vacías, más adelante convertiremos esta columna en números para el modelo.
    df["chiefcomplaint"] = df["chiefcomplaint"].fillna("")
    
    ruta_csv_edstays = "/opt/airflow/plugins/triage_ed_limpio.csv"
    df.to_csv(ruta_csv_edstays, index=False) # Recomendado: index=False para evitar la columna Unnamed: 0
        
    return ruta_csv_edstays
    
    
    