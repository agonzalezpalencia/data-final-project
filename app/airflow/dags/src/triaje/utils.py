import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import mlflow

def enrutador_procesado (**kwargs):
    # Recuperamos el params, usamos un valor vacío por defecto para evitar errores de comparación
    opciones_elegidas = kwargs['params'].get('modelos_a_entrenar', '')

    tareas_siguientes =[]

    # Se realizará el primer procesado para RF o TF
    if 'RF' in opciones_elegidas or 'TF' in opciones_elegidas:
        tareas_siguientes.append("procesar_dataset")
        print("Se realizará el procesado para Tensorflow o RandomForest")

    # Se realizará el segundo procesado para LR
    if 'LR' in opciones_elegidas:
        tareas_siguientes.append("procesar_dataset_edstays")
        print("Se realizará el procesado para LogisticRegression")

    # Si no se selecciona ningún modelo o se escribe mal se controla el abandonar el flujo
    if not tareas_siguientes:
        tareas_siguientes.append("abandonar_flujo")
        print("No se ha introducido ningún modelo a entrenar... Saliendo")

    return tareas_siguientes

def enrutador_modelos (**kwargs):
    # Recuperamos el params, usamos un valor vacío por defecto para evitar errores de comparación
    opciones_elegidas = kwargs['params'].get('modelos_a_entrenar', '')

    tareas_siguientes =[]

    # Se realizará el primer procesado para RF o TF
    if 'RF' in opciones_elegidas:
        tareas_siguientes.append("entrenar_modelo_rf")
        print("Se procede al entrenamiento del RandomForest")

    # Se realizará el segundo procesado para LR
    if 'TF' in opciones_elegidas:
        tareas_siguientes.append("entrenar_modelo_tf")
        print("Se procede al entrenamiento del modelo de Tensorflow")

    # No hay descarte del flujo ya que no llega aquí si no se selecciona RF o TF

    return tareas_siguientes

def plot_keras_history(history):
    # Método para dibujar gráfica de loss y accuracy en modelos
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Gráfica de pérdida
    axes[0].plot(history.history['loss'], label='Train Loss', color='blue')
    axes[0].plot(history.history['val_loss'], label='Val Loss', color='red')
    axes[0].set_title('Pérdida (Loss)')
    axes[0].set_xlabel('Épocas')
    axes[0].legend()
    axes[0].grid(True)

    # Gráfica de accuracy
    axes[1].plot(history.history['binary_accuracy'], label='Train Accuracy', color='blue')
    axes[1].plot(history.history['val_binary_accuracy'], label='Val Accuracy', color='green')
    axes[1].set_title('Precisión (Accuracy)')
    axes[1].set_xlabel('Épocas')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    path_keras = "/tmp/keras_history.png"
    plt.savefig(path_keras)
    plt.close()

    # Guardar en los artefactos de MLflow
    mlflow.log_artifact(path_keras, artifact_path="graficos")


def plot_matriz_confusion(y_true, y_pred, nombre_modelo):
    # Calculamos la matriz con scikit-learn
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Urgente', 'Urgente'])
    disp.plot(ax=ax, cmap='Blues', colorbar=False, values_format='d')
    
    plt.title(f'Matriz de Confusión - {nombre_modelo}')
    plt.tight_layout()

    path_cm = f"/tmp/cm_{nombre_modelo}.png"
    plt.savefig(path_cm)
    plt.close()

    # Subimos la imagen a los artefactos de MLflow
    mlflow.log_artifact(path_cm, artifact_path="graficos")
    
# Función para graficar los resultados del GridSearchCV
def plot_grid_search_results(res, best_model_index):
    accuracy_list = res["mean_test_accuracy"]
    loss_list = -res["mean_test_neg_log_loss"]
    model_indices = range(len(res))
        
    # Obtener índice del mejor modelo de los 42 según (refit:'neg_log_loss')
    # best_model_index = grid.best_index_

    # Gráfico
    plt.figure(figsize=(15,5))

    # Gráfico 1: Comparando pérdida entre modelos
    plt.subplot(1,2,1)
    plt.plot(model_indices, loss_list, color='blue')
    plt.plot(best_model_index, loss_list[best_model_index], marker=".", markersize=15, color="gold", markeredgecolor="black", label="Mejor modelo")
    plt.xlabel("Combinaciones de Hiperparámetros")
    plt.ylabel("Loss (Log-Loss)")
    plt.title("Comparación de pérdida (Loss) entre modelos")
    plt.legend()

    # Gráfico 2: Comparando precisión entre modelos
    plt.subplot(1,2,2)
    plt.plot(model_indices, accuracy_list, color='orange')
    plt.plot(best_model_index, accuracy_list[best_model_index], marker=".", markersize=15, color="gold", markeredgecolor="black", label="Mejor modelo")
    plt.xlabel("Combinaciones de Hiperparámetros")
    plt.ylabel("Precisión (Accuracy)")
    plt.title("Comparación de precisión (Accuracy) entre modelos")
    plt.legend()

    plt.tight_layout()
    
    path_grid = f"/tmp/grid_search_comparative.png"
    plt.savefig(path_grid)
    plt.close()
    
    # Subir a artefactos de MLflow
    mlflow.log_artifact(path_grid, artifact_path="graficos")