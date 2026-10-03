import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import mlflow

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