import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from src.triaje.utils import plot_keras_history, plot_matriz_confusion

def entrenar_modelo_tf(ti):
    # Métricas del sistema
    mlflow.enable_system_metrics_logging()
    csv_limpio = ti.xcom_pull(task_ids="procesar_dataset")
    df = pd.read_csv(csv_limpio)

    X = df[['temperature', 'heartrate','resprate', 'o2sat', 'sbp', 'dbp','pain']].astype({
        'temperature': 'float32', 'heartrate': 'float32', 'resprate': 'float32',
        'o2sat': 'float32', 'sbp': 'float32', 'dbp': 'float32', 'pain': 'float32'
    }).to_numpy()

    y = df[['acuity']].to_numpy(dtype=np.float32).ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    normalizator = tf.keras.layers.Normalization(axis=-1)
    tf.keras.utils.set_random_seed(42)
    normalizator.adapt(X_train)

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(7,)),
        normalizator,
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', patience=3, restore_best_weights=True)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=0.01)
    loss = tf.keras.losses.BinaryCrossentropy()
    metric = tf.keras.metrics.BinaryAccuracy()

    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment('Experimento Entreno')

    with mlflow.start_run(run_name="Keras_Binary") as run_tf:
        mlflow.tensorflow.autolog()
        history = model.fit(X_train, y_train, epochs=25, batch_size=8, callbacks=[early_stopping], validation_data=(X_test, y_test))
        # Función para dibujar las gráficas y guardar el artefacto
        plot_keras_history(history)
        accuracy_tf = history.history['val_binary_accuracy'][-1]
        # Obtenemos las predicciones sobre el modelo para la matriz de confunsion
        y_pred_probs = model.predict(X_test)
        # Como usamos una sigmoide (0 a 1), convertimos la probabilidad en 0 o 1 usando un umbral de 0.5
        y_pred = (y_pred_probs >= 0.5).astype(int).ravel()
        # Llamada al método para crear un artefacto con la matrix de confusión
        # El primer parámetro hace referencia a los resultados obtenidos en el test, 
        # el segundo a la predicciones, 
        # el tercero es el nombre del modelo para dibujar en la gráfica
        plot_matriz_confusion(y_test, y_pred, "Keras_Red_Neuronal", display_labels=['No urgente', 'Urgente'])
        id_tf = run_tf.info.run_id
        print("Accuracy Tensorflow:", accuracy_tf)

    model_uri = f"runs:/{id_tf}/model"
    mlflow.register_model(model_uri=model_uri, name="Clasificador_TF_Triaje")

def entrenar_modelo_rf(ti):
    csv_limpio = ti.xcom_pull(task_ids="procesar_dataset")
    df = pd.read_csv(csv_limpio)

    X = df[['temperature', 'heartrate','resprate', 'o2sat', 'sbp', 'dbp','pain']].astype({
        'temperature': 'float32', 'heartrate': 'float32', 'resprate': 'float32',
        'o2sat': 'float32', 'sbp': 'float32', 'dbp': 'float32', 'pain': 'float32'
    }).to_numpy()

    y = df[['acuity']].to_numpy(dtype=np.float32).ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment('Experimento Entreno')

    with mlflow.start_run(run_name="RandomForest_Base") as run_rf:
        mlflow.sklearn.autolog()
        rf.fit(X_train, y_train)
        accuracy_rf = rf.score(X_test, y_test)
        # Predicciones para realizar la matriz de confusión
        y_pred = rf.predict(X_test)
        # Llamada al método para crear un artefacto con la matrix de confusión
        # El primer parámetro hace referencia a los resultados obtenidos en el test, 
        # el segundo a la predicciones, 
        # el tercero es el nombre del modelo para dibujar en la gráfica
        plot_matriz_confusion(y_test, y_pred, "RandomForest", display_labels=['No urgente', 'Urgente'])
        id_rf = run_rf.info.run_id
        print("Accuracy RandomForest:", accuracy_rf)

    model_uri = f"runs:/{id_rf}/model"
    mlflow.register_model(model_uri=model_uri, name="Clasificador_RF_Triaje")