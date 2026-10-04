import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, accuracy_score
from sklearn.metrics import log_loss
import pandas as pd

def sepsis_rd_train(ti):

    ruta_tr, ruta_val, ruta_test = ti.xcom_pull(task_ids="separar_train_validation_test")
    tr = pd.read_parquet(ruta_tr)
    val = pd.read_parquet(ruta_val)
    test = pd.read_parquet(ruta_test)

    X_tr = tr.drop(columns=['SepsisLabel']).to_numpy()
    y_tr = tr['SepsisLabel'].to_numpy()
    X_test = val.drop(columns=['SepsisLabel']).to_numpy()
    y_test = val['SepsisLabel'].to_numpy()
    X_val = test.drop(columns=['SepsisLabel']).to_numpy()
    y_val = test['SepsisLabel'].to_numpy()

    rf = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=50,
        max_samples=0.2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment('Experimento Entreno')

    with mlflow.start_run(run_name="RandomForest_Base") as run_rf:
        
        rf.fit(X_tr, y_tr)
        mlflow.sklearn.log_model(rf, "model")

        pred_val = rf.predict_proba(X_val)[:, 1]
        pred_rf = rf.predict_proba(X_test)[:, 1]

        print("AUPRC validación:", average_precision_score(y_val, pred_val))
        print("AUPRC test:", average_precision_score(y_test, pred_rf))
        print("Pérdida validación:", log_loss(y_val, pred_val))
        print("Pérdida test:", log_loss(y_test, pred_rf))
        print("Accuracy validación:", accuracy_score(y_val, pred_val > 0.5))
        print("Accuracy test:", accuracy_score(y_test, pred_rf > 0.5))

        id_rf = run_rf.info.run_id

    model_uri = f"runs:/{id_rf}/model"
    mlflow.register_model(model_uri=model_uri, name="Clasificador_RF_Sepsis")