import pandas as pd
import numpy as np

def entrenar_modelo_rf_m(ti):
    # Lazy Import: evita OOM de memoria en Airflow
    import mlflow
    import mlflow.sklearn
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import log_loss
    from src.triaje_version_m.data_processing_m import build_features
    
    csv_limpio = ti.xcom_pull(task_ids="procesar_dataset")
    df = pd.read_csv(csv_limpio)
    
    # definir target (0=acuity 1-2, 1=acuity 3-5)
    y = (df['acuity'] < 3).astype(int).to_numpy()
  
    # split y entrenamiento
    df_train, df_test, y_train, y_test = train_test_split(
        df, 
        y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )

    Xtr, med = build_features(df_train)
    Xte, _   = build_features(df_test, medians=med)
    
    # RESULTADO Test loss: 0.635  Test accuracy: 0.684
    rf = RandomForestClassifier(
        n_estimators=300, 
        max_depth=5,
        min_samples_leaf=2, 
        random_state=42,
        max_features='sqrt',
        max_samples=0.2,
        oob_score=True
    )

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment('Entreno Random Forest Triaje M')

    with mlflow.start_run(run_name="RandomForest_m") as run_rf:
        mlflow.sklearn.autolog() # autoregistro de méticas, parámetros y modelo (artefactos mlflow)
        rf.fit(Xtr, y_train)
        
        id_rf = run_rf.info.run_id # caputar ID del Run
        
        print(f"Columnas: {Xtr.shape[1]}  Accuracy: {rf.score(Xte, y_test):.3f}")

        test_accuracy = rf.score(Xte, y_test)
        test_loss = log_loss(y_test, rf.predict_proba(Xte))
        print(f"Test loss: {test_loss:.3f}  Test accuracy: {test_accuracy:.3f}")
        
        
        # importancia relativa de cada variable dentro del modelo una vez entrenado
        imp = pd.Series(rf.feature_importances_, index=Xtr.columns).sort_values(ascending=False)
        print("Top 15 features más importantes del modelo:")
        print(imp.head(15))
        
    model_uri = f"runs:/{id_rf}/model"
    mlflow.register_model(model_uri=model_uri, name="Clasificador_RF_Triaje_m") # coger artefacto y registrarlo en el Model Registry de mlflow para control de versiones
    