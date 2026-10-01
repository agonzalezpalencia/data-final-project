# Extiende la imagen oficial de Airflow con las librerías que necesitan los DAGs
# reales del proyecto (MLflow, scikit-learn, TensorFlow, pandas, y los loaders
# de datasets ucimlrepo/wfdb). Sin esto, los workers solo pueden ejecutar los
# DAGs de demostración con time.sleep(), no entrenar modelos reales.
FROM apache/airflow:2.11.2

COPY requirements-mlops.txt /requirements-mlops.txt

RUN pip install --no-cache-dir -r /requirements-mlops.txt
