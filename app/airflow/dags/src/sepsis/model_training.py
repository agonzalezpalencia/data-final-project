import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, accuracy_score
from sklearn.metrics import log_loss
from mlflow.models import infer_signature
import pandas as pd

# Método que permite realizar el entrenamiento del modelo de Sepsis usando el algoritmo de RandomForest de Scikit-Learn
def sepsis_rd_train(ti):

    # Recuperamos el dataset de train, validation y de test desde la tarea anterior
    ruta_tr, ruta_val, ruta_test = ti.xcom_pull(task_ids="separar_train_validation_test")
    tr = pd.read_parquet(ruta_tr)
    val = pd.read_parquet(ruta_val)
    test = pd.read_parquet(ruta_test)

    # En los datos de entrada eliminamos la etiqueta de SepsisLabel ya que es nuestra etiqueta, en los datos que suponen las etiquetas utilizamos la etiqueta SepsisLabel únicamente
    X_tr = tr.drop(columns=['SepsisLabel']).to_numpy()
    y_tr = tr['SepsisLabel'].to_numpy()
    X_test = val.drop(columns=['SepsisLabel']).to_numpy()
    y_test = val['SepsisLabel'].to_numpy()
    X_val = test.drop(columns=['SepsisLabel']).to_numpy()
    y_val = test['SepsisLabel'].to_numpy()

    # Utilizamos para entrenar al modelo el algoritmo de RandomForest que nos proporciona Scikit-Learn, lo que hará será entrenar todos los árboles del bosque con una muestra aleatoria de
    # los datos, asñi no se entrenan todos con los mismos datos
    rf = RandomForestClassifier(
        n_estimators=200, # Número de árboles que tendrá el bosque
        min_samples_leaf=50, # Cantidad mínima de registros que tendrá cada hoja
        max_samples=0.2, # Cada árbol usará una muestra aleatoria del 20% de las filas
        class_weight="balanced_subsample", # Pesos utilizados para el entrenamiento, utilizamos el "balanced_subsample" y a medida que el entrenamiento avance, se irán cambiando
        n_jobs=-1, # Hacemos que no use todos los núcleos de la CPU para el entrenamiento
        random_state=42, # Semilla 42 por defecto 
    )

    # Activamos MLFlow para subir el modelo
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment('Experimento Sepsis')

    # Especificamos que utilizaremos el algortimo RandomForest para el entrenamiento
    with mlflow.start_run(run_name="RandomForest_Base") as run_rf:
        
        rf.fit(X_tr, y_tr) # Realizamos el entrenamiento
        y_pred = pd.Series(rf.predict(X_tr), name="SepsisLabel") # Guardamos las predicciones que ha hecho el modelo
        signature = infer_signature(X_tr, y_pred) # Creamos la firma con los datos de entrada y salida en MLFlow

        # Usamos el log de sklearn que nos proporciona MLFlow para poder poner los datos de entrada y salida del modelo registrados en MLFlow
        mlflow.sklearn.log_model(
            sk_model=rf, 
            name="model",
            signature=signature,
            input_example=X_tr[:5],
            registered_model_name="Clasificador_RF_Sepsis"
        ) 

        # Realizamos predicciones sobre los datos de validación y de test
        pred_val = rf.predict_proba(X_val)[:, 1]
        pred_rf = rf.predict_proba(X_test)[:, 1]

        # Obtenemos métricas del modelo a partir de los datos de validación y test extraídos
        print("AUPRC validación:", average_precision_score(y_val, pred_val))
        print("AUPRC test:", average_precision_score(y_test, pred_rf))
        print("Pérdida validación:", log_loss(y_val, pred_val))
        print("Pérdida test:", log_loss(y_test, pred_rf))

        # Usamos un umbral de 0.5 para la precisión debido ya que es el valor por defecto y un punto neutral:
        #   Si bajamos el umbral, incrementaríamos la detección de sepsis pero con muchos falsos positivos
        #   Si aumentamos el umbral, bajaríamos la detección de sepsis, pero habría muchos casos que son positivos reales que nos saltaríamos
        print("Accuracy validación:", accuracy_score(y_val, pred_val > 0.5)) 
        print("Accuracy test:", accuracy_score(y_test, pred_rf > 0.5))

        # Recogemos el id para pasarlo más adelante a MLFlow y que detecte la subida de un nuevo modelo
        id_rf = run_rf.info.run_id

    # Registramos el modelo
    model_uri = f"runs:/{id_rf}/model"
    mlflow.register_model(model_uri=model_uri, name="Clasificador_RF_Sepsis")