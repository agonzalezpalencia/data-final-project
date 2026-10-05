# Proyecto final: Datasets de Triaje y Sepsis combinado con Airflow y MLflow

Proyecto de MLOps sobre datos sanitarios. Se entrenan modelos de clasificación para dos casos de uso: los flujos de limpieza y entrenamiento se orquestan con Airflow y los modelos resultantes se versionan en MLflow. Jupyter se usa para la exploración y las pruebas previas.

| Caso | Objetivo | Dataset | Modelos |
|---|---|---|---|
| Triaje | Predecir si un paciente que llega a urgencias es urgente o no | [MIMIC-IV-ED Demo 2.2](https://physionet.org/content/mimic-iv-ed-demo/2.2/) | Red neuronal (Keras), Random Forest y Regresión Logística |
| Sepsis | Predecir si un paciente presenta sepsis en cada registro horario de su estancia | [PhysioNet Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/) | Random Forest |

La documentación detallada de cada caso se encuentra en [README_triaje.md](docs/README_triaje.md) y [README_sepsis.md](docs/README_sepsis.md).

## Estructura

```text
app/
  airflow/          Punto de entrada de Airflow, DAGs en dags/ y su código en dags/src/
  mlflow/           Servidor de MLFlow
  jupyter/          Cuadernos utilizados para exploración y pruebas sobre los Datasets
docs/               Documentación sobre filtrado de datasets y entrenamiento de modelos
```

## Requisitos

- Docker Desktop instalado.
- Datos de entrada de cada DAG:

| Dataset | Fichero | Ubicación |
|---|---|---|
| Triaje | `triage.csv.gz` y `edstays.csv.gz` | `app/airflow/datos/mimic-iv-ed-demo-2.2/mimic-iv-ed-demo-2.2/ed/` |
| Sepsis | `sepsis_parquet_raw.parquet` | `app/airflow/parquets/` |

El parquet de sepsis se genera desde el cuaderno de Jupyter [notebook_abel_sepsis.ipynb](app/jupyter/notebooks/notebook_abel_sepsis.ipynb), que une todos los ficheros `.psv` (pacientes) de las carpetas `training_setA` y `training_setB` en un único fichero `.parquet`.

## Ejecución

### 1. Levantar MLflow

Debe arrancarse primero, porque es el encargado de crerar la red de Docker `mlflow-net` que usa Airflow para conectarse y subir los modelos.

```bash
cd app/mlflow
docker compose up -d
```

Puedes utilizarlo en: http://localhost:5000

### 2. Levantar Airflow

```bash
cd app/airflow
docker compose up -d
```

La primera vez se construye la imagen con las dependencias de ML, por lo que tardará un poco en levantarse completamente.

Puedes utilizarlo en: http://localhost:8080

### 3. Lanzar los DAGs

Los DAGs que entrenan los modelos son dos:

| Fichero | DAG en Airflow | Qué hace |
|---|---|---|
| `dag_triaje.py` | `Triaje_DAG` | Lee y limpia el dataset de triaje y entrena en paralelo los tres modelos (Keras, LogisticRegression y RandomForest) |
| `dag_sepsis.py` | `sepsis_dag` | Carga, limpia y particiona el dataset de sepsis y entrena el modelo usando el algoritmo Random Forest |

### 4. Consultar los resultados en MLFlow

Cada ejecución crea una ejecución nueva en el experimento y una versión nueva de cada modelo registrado.

| Caso | Experimento | Modelos registrados |
|---|---|---|
| Triaje | `Experimento Entreno` | `Clasificador_TF_Triaje`, `Clasificador_RF_Triaje`, `Clasificador_LR_Triaje` |
| Sepsis | `Experimento Sepsis` | `Clasificador_RF_Sepsis` |

### 5. Parar los servicios

```bash
docker compose down
```

Se ejecuta en `app/airflow` y después en `app/mlflow`.

## Jupyter (No requerido)

Con MLFlow ya levantado:

```bash
cd app/jupyter
docker compose up -d
```

Puedes utilizarlo en: http://localhost:8888

---