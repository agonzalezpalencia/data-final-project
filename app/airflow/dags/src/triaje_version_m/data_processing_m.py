import pandas as pd
import numpy as np
import os

 # diccionario para agrupar motivo de la consulta de la columna "chiefcomplaint"
# [sub] son palabras literales 
# [exact] son siglas que cuentan como una palabra completa
CONCEPTS = {
    'cc_arrest':   {'sub': ['arrest', 'intubat'], 'exact': []},
    'cc_neuro':    {'sub': ['droop', 'numb', 'weakness', 'dizz', 'diplopia', 'syncope', 'seizure',
                            'confusion', 'altered', 'letha', 'stroke', 'headache', 'hallucin',
                            'ambulate', 'tinnitus', 'head injury', 'head bleed'],
                    'exact': ['ams', 'cva', 'ich', 'sdh', 'sah']},
    'cc_cardiac':  {'sub': ['chest', 'palpitation', 'tachycardia', 'nstemi', 'aortic', 'atrial',
                            'ekg', 'hypertension', 'hypotension'], 'exact': []},
    'cc_resp':     {'sub': ['dyspnea', 'shortness', 'cough', 'hypox', 'laryngitis'],
                    'exact': ['sob', 'pe']},
    'cc_bleed':    {'sub': ['bleed', 'brbpr', 'hematemesis', 'hematuria', 'vomiting blood', 'emesis'],
                    'exact': []},
    'cc_metab':    {'sub': ['abnormal', 'lab', 'hyperglyc', 'hypoglyc', 'anemia', 'dehydrat',
                            'fatigue', 'hypotherm', 'neutropenia'],
                    'exact': ['dka', 'inr']},
    'cc_abd':      {'sub': ['abd', 'abdominal', 'n/v', 'nausea', 'vomit', 'diarrhea', 'epigastric',
                            'suprapubic', 'rlq', 'luq', 'ascites', 'rectal', 'urinary', 'dysuria',
                            'foley'], 'exact': []},
    'cc_infect':   {'sub': ['infect', 'fever', 'cellulitis', 'ulcer', 'abscess'],
                    'exact': ['ili']},
    'cc_trauma':   {'sub': ['fall', 'mvc', 'car vs', 'assault', 'injur', 'fracture', 'laceration',
                            'wound', 'trauma'], 'exact': ['fx']},
    'cc_psych':    {'sub': ['depress', 'anxiety', 'psych', 'etoh', 'overdose', 'insomnia'],
                    'exact': ['si']},
    'cc_vascular': {'sub': ['swelling'], 'exact': ['dvt']},
    'cc_localpain': {'sub': ['back', 'foot', 'knee', 'hand', 'shoulder', 'wrist', 'extremity'],
                    'exact': ['leg', 'arm', 'toe', 'rib', 'hip', 'ear', 'eye']},
    'cc_minor':    {'sub': ['eval', 'refill', 'picc', 'tube', 'hemodialysis', 'rash', 'allergic'],
                    'exact': []},
    'cc_transfer': {'sub': ['transfer'], 'exact': []},
    'cc_critical': {'sub': ['arrest', 'unresponsive', 'stroke', 'chest pain', 'shortness',
                            'seizure', 'bleeding'], 'exact': ['sob']},
}

 # featuring 
VITALS = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain']

# ---------------------------------------------------
# FUNCIONES
# ---------------------------------------------------

# devuelve 1 si el texto contiene alguna palabra de [sub] o [exact] y 0 si no contiene
def has_concept(text, sub, exact):
    words = set(text.replace(',', ' ').replace('/', ' ').replace(';', ' ')
                    .replace('?', ' ').replace('-', ' ').replace('.', ' ').split())
    return int(any(k in text for k in sub) or any(w in words for w in exact))



# convertir el dataframe en una tabla numérica para darsela al modelo
# limpiar de nulos
# rellenar huecos de variables con medias
def build_features(raw, medians=None, use_cc=True):
    d = raw.copy()

    for c in VITALS:
        d[c] = pd.to_numeric(d[c], errors='coerce')

    if medians is None:
        medians = d[VITALS].median()

    out = pd.DataFrame(index=d.index)
    out['pain_missing'] = d['pain'].isna().astype(int)   # antes de rellenar
    d['pain'] = d['pain'].fillna(0)
    d[VITALS] = d[VITALS].fillna(medians)

    for c in VITALS:
        out[c] = d[c]

    # Derivadas (temperatura en °F)
    shock = d['heartrate'] / d['sbp'].replace(0, np.nan)
    # feature derivada de recuencia cardíaca / presión sistólica
    out['feat_shock_index'] = shock.fillna(medians['heartrate'] / medians['sbp'])
    # feature derivada que suma sbp y resprate
    out['feat_qsofa_score'] = (d['sbp'] <= 100).astype(int) + (d['resprate'] >= 22).astype(int)

    out['flag_fever']       = (d['temperature'] >= 100.4).astype(int)
    out['flag_hypothermia'] = (d['temperature'] < 95.0).astype(int)
    out['flag_hypoxia']     = (d['o2sat'] < 92.0).astype(int)
    out['flag_tachycardia'] = (d['heartrate'] > 100.0).astype(int)
    out['flag_bradycardia'] = (d['heartrate'] < 60.0).astype(int)
    out['flag_tachypnea']   = (d['resprate'] >= 22.0).astype(int)
    out['flag_hypotension'] = (d['sbp'] < 90.0).astype(int)
    out['flag_severe_pain'] = (d['pain'] >= 7.0).astype(int)

    # Conceptos del chiefcomplaint
    if use_cc:
        txt = d['chiefcomplaint'].fillna('').str.lower()
        for col, spec in CONCEPTS.items():
            out[col] = txt.apply(lambda t: has_concept(t, spec['sub'], spec['exact']))
        out['cc_unknown'] = txt.apply(lambda t: int(t == '' or 'unknown' in t or t == 'sw'))
        out['feat_cc_n_symptoms'] = txt.apply(lambda t: t.count(',') + t.count(';') + 1)

    return out.astype('float32'), medians



def leer_dataset(ti):
    csv_base = "/opt/airflow/datos/mimic-iv-ed-demo-2.2/mimic-iv-ed-demo-2.2/ed/triage.csv.gz"
    
    # Comprobar existencia del archivo csv
    if not os.path.exists(csv_base):
        raise FileNotFoundError(f"No se encontró el archivo en {csv_base}")
    
    # Leer 5 filas del archivo para comprobar columnas sin saturar memoria RAM low_tier_tasks
    triaje_sample = pd.read_csv(csv_base, nrows=5)
    print(f"Archivo verificado. Columnas: {triaje_sample.shape[1]}")

    return csv_base

def procesar_dataset(ti):
    csv_base = ti.xcom_pull(task_ids="leer_dataset")
    triaje = pd.read_csv(csv_base)
    
    df = triaje[triaje['acuity'].notna() & triaje['pain'].notna() &
                triaje['temperature'].notna() & triaje['heartrate'].notna() &
                triaje['o2sat'].notna()].drop(columns=['subject_id', 'stay_id']).copy()

    # # 0 = acuity 1-2, 1 = acuity 3-5 
    # y = (df['acuity'] < 3).astype(int).to_numpy()
    # print(np.unique(y, return_counts=True))

    # print(df[:40])

    print("Procesando dataset: eliminar nulos y crear features derivadas")
    
    ruta_csv_limpio = "/opt/airflow/plugins/triage_limpio_version_m.csv"
    df.to_csv(ruta_csv_limpio, index=False) # Recomendado: index=False para evitar la columna Unnamed: 0

    return ruta_csv_limpio