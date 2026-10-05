import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from src.utils import plot_keras_history, plot_matriz_confusion, plot_grid_search_results
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def entrenar_modelo_tf(ti, epochs=25, neur=32, learning_rate=0.01, run_name="Keras_Binary"):
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
        tf.keras.layers.Dense(neur, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', patience=3, restore_best_weights=True)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)
    loss = tf.keras.losses.BinaryCrossentropy()
    metric = tf.keras.metrics.BinaryAccuracy()

    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment('Experimento Entreno')

    with mlflow.start_run(run_name=run_name) as run_tf:
        mlflow.tensorflow.autolog()
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=8, callbacks=[early_stopping], validation_data=(X_test, y_test))
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
        plot_matriz_confusion(y_test, y_pred, "Keras_Red_Neuronal")
        id_tf = run_tf.info.run_id
        print("Accuracy Tensorflow:", accuracy_tf)

    model_uri = f"runs:/{id_tf}/model"
    mlflow.register_model(model_uri=model_uri, name="Clasificador_TF_Triaje")
    
    return float(accuracy_tf) # devolver accuracy como float para que Airflow lo maneje en XCom

def entrenar_modelo_rf(ti, n_estimators=100, random_state=42, run_name="RandomForest_Base"):
    csv_limpio = ti.xcom_pull(task_ids="procesar_dataset")
    df = pd.read_csv(csv_limpio)

    X = df[['temperature', 'heartrate','resprate', 'o2sat', 'sbp', 'dbp','pain']].astype({
        'temperature': 'float32', 'heartrate': 'float32', 'resprate': 'float32',
        'o2sat': 'float32', 'sbp': 'float32', 'dbp': 'float32', 'pain': 'float32'
    }).to_numpy()

    y = df[['acuity']].to_numpy(dtype=np.float32).ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment('Experimento Entreno')

    with mlflow.start_run(run_name=run_name) as run_rf:
        mlflow.sklearn.autolog()
        rf.fit(X_train, y_train)
        accuracy_rf = rf.score(X_test, y_test)
        # Predicciones para realizar la matriz de confusión
        y_pred = rf.predict(X_test)
        # Llamada al método para crear un artefacto con la matrix de confusión
        # El primer parámetro hace referencia a los resultados obtenidos en el test, 
        # el segundo a la predicciones, 
        # el tercero es el nombre del modelo para dibujar en la gráfica
        plot_matriz_confusion(y_test, y_pred, "RandomForest")
        id_rf = run_rf.info.run_id
        print("Accuracy RandomForest:", accuracy_rf)

    model_uri = f"runs:/{id_rf}/model"
    mlflow.register_model(model_uri=model_uri, name="Clasificador_RF_Triaje")
    
    return float(accuracy_rf) # devolver accuracy como float para que Airflow lo maneje en XCom
    
def entrenar_modelo_lr(ti, c_values=(0.01, 0.03, 0.1, 0.3, 1, 3, 10), run_name="LogisticRegression_GridSearch"):
    mlflow.enable_system_metrics_logging()
    csv_limpio = ti.xcom_pull(task_ids="procesar_dataset_edstays")
    df = pd.read_csv(csv_limpio)
    
    # Declaramos las columnas númericas 
    numerical_columns = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain", "pain_not_assessable", "shock_index", "pulse_pressure", "arrival_hour"]

    # Definimos las observaciones
    X = df[numerical_columns + ["arrival_transport", "chiefcomplaint"]]

    # Definimos las etiquetas o los valores que tiene que predecir el modelo (Hacemos que 1 y 2 se normaicen a 0 y 3,4,5 se normalicen a 1)
    y = (df["acuity"] <= 2).astype(int)
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    # Declaramos la preparación que haremos sobre los datos
    prep = make_column_transformer(
        # Con SimpleImputer rellenamos los huecos que no contengan números para no sesgar los datos
        (make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), numerical_columns),
        # Como en `arrival_transport` separamos en categorías, OneHotEncoder nos ayuda a separar las cateorías y transformarlas a números
        (OneHotEncoder(handle_unknown="ignore"), ["arrival_transport"]),
        # Normalizamos la columna de 'chiefcomplaint' estableciendo pesos según las repeticiones de las palabras
        (TfidfVectorizer(ngram_range=(1,2), min_df=2), "chiefcomplaint")
    )
    
    # Declaramos el modelo que vamos a entrenar utilizando los datos que hemos preparado en los pasos anteriores, además utilizamos el algoritmo
    # de LogisticRegression con un máximo de 5000 iteraciones sobre nuestros datos
    model = make_pipeline(prep, LogisticRegression(max_iter=5000))

    # Usamos RepeatedStratifiedKFold para recorrer los datos de múltiples maneras distintas y evaluar el modelo con mayor fiabilidad.
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

    # Utilizamos GridSearchCV para poder especificar el modelo que vamos a entrenar y los hiperparámetros que vamos a utilizar
    grid = GridSearchCV(
        model,
        {
            "logisticregression__C": list(c_values),
            "logisticregression__class_weight": ["balanced", None],
            "columntransformer__tfidfvectorizer__min_df": [1, 2, 3],
            # Especificamos las flags que vamos a usar a la hora de usar GridSearchCV
        }, 
        cv=cv, 
        scoring=["accuracy", "neg_log_loss"], 
        n_jobs=2, # evitar saturación del worker 
        refit='neg_log_loss', 
        verbose=3, 
        return_train_score=True
    )
    
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("Experimento Entreno")
    
    
    with mlflow.start_run(run_name=run_name) as run_lr:
        # mlflow.sklearn.autolog()
        # Entrenamiento del modelo
        grid.fit(X_train,y_train)
        
        res_df = pd.DataFrame(grid.cv_results_)
        
        # Guardar artefactos visuales
        plot_grid_search_results(res_df, grid.best_index_)
        
        y_pred = grid.predict(X_test)
        plot_matriz_confusion(y_test, y_pred, "LogisticRegression")
        
        accuracy_lr = grid.score(X_test, y_test)
        mlflow.log_metric("accuracy_test", accuracy_lr)
        mlflow.log_params(grid.best_params_)
        
        # Guardar el mejor modelo en la carpeta "model" en MLflow
        mlflow.sklearn.log_model(grid.best_estimator_, artifact_path="model")
        
        id_lr = run_lr.info.run_id
        print("Accuracy de Logistic Regression: ", accuracy_lr)
        
    model_uri = f"runs:/{id_lr}/model"
    mlflow.register_model(model_uri=model_uri, name="Clasificador_LR_Triaje")
    
    return float(accuracy_lr) # devolver accuracy como float para que Airflow lo maneje en XCom
        