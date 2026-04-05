import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "datos" / "clean_data"
MODEL_DIR = BASE_DIR / "datos" / "modelos"

def perform_training():
    """Lógica central de construcción del modelo"""
    # 1. Carga y preparación
    datasets = []
    for f in ["processed_spanish.csv", "processed_english.csv"]:
        path = DATA_DIR / f
        if path.exists():
            tmp = pd.read_csv(path)
            datasets.append(pd.DataFrame({'text': tmp['pros_clean'].dropna(), 'label': 1}))
            datasets.append(pd.DataFrame({'text': tmp['cons_clean'].dropna(), 'label': 0}))
    
    full_data = pd.concat(datasets).reset_index(drop=True)
    X_train, X_test, y_train, y_test = train_test_split(
        full_data['text'], full_data['label'], test_size=0.2, random_state=42, stratify=full_data['label']
    )

    # 2. Vectorización 
    vectorizer = TfidfVectorizer(max_features=50, ngram_range=(1, 1))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 3. Entrenamiento
    model = LogisticRegression(C=1.0, random_state=42, class_weight='balanced')
    model.fit(X_train_tfidf, y_train)

    # 4. Cálculo de métricas
    y_pred = model.predict(X_test_tfidf)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred)
    }

    return model, vectorizer, metrics, X_test, y_test

def save_local_resources(model, vectorizer, metrics, total_samples):
    """Guarda los archivos físicos y metadata localmente"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "sentiment_model_global.pkl")
    joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer_global.pkl")
    
    # Generación de metadata 
    log_path = MODEL_DIR / "classification_metadata.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Construction model: Logistic Classifier\n")
        f.write(f"Training date: {datetime.now()}\n")
        f.write(f"Total Samples Trained: {total_samples}\n")

if __name__ == "__main__":
    print("Ejecutando entrenamiento local...")
    model, vec, met, _, _ = perform_training()
    save_local_resources(model, vec, met, 50) 
    print(f"Modelo guardado. Accuracy: {met['accuracy']:.2f}")