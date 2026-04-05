import mlflow
import mlflow.sklearn
from training import perform_training, save_local_resources
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Configuración de rutas ---
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "datos" / "modelos"

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Glassdoor_Sentiment_Analysis")

def run_mlops_pipeline():
    with mlflow.start_run():
        # Llamado al archivo de entrenamiento
        model, vectorizer, metrics, X_test, y_test = perform_training()
        
        # Guardamos localmente resultados
        save_local_resources(model, vectorizer, metrics, len(X_test)*5)

        # Registro en MLflow
        mlflow.log_param("max_features", 50)
        mlflow.log_param("ngram_range", "(1,1)")
        mlflow.log_param("model_type", "Logistic Regression")

        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        
        mlflow.sklearn.log_model(model, "sentiment_model")

        if MODEL_DIR.exists():
            mlflow.log_artifacts(str(MODEL_DIR), artifact_path="modelos_y_metadata")
        
        # Generar y loguear Matriz de Confusión
        y_pred = model.predict(vectorizer.transform(X_test))
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Cons', 'Pros'], yticklabels=['Cons', 'Pros'], ax=ax)
        ax.set_title('Confusion Matrix')
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)
        
    print("Pipeline de MLOps finalizado y registrado.")

if __name__ == "__main__":
    run_mlops_pipeline()