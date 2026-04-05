import pandas as pd
import joblib
from pathlib import Path
from pysentimiento import create_analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- Configuración de rutas ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "datos"  / "clean_data"
MODEL_DIR = BASE_DIR / "datos" / "modelos"

def load_resources():
    """Carga el modelo, vectorizador y analizadores (Pysentimiento y VADER)"""
    try:
        model_path = MODEL_DIR / "sentiment_model_global.pkl"
        vec_path = MODEL_DIR / "tfidf_vectorizer_global.pkl"
        
        # Validación de parametros
        if not model_path.exists() or not vec_path.exists():
            raise FileNotFoundError("Los archivos del modelo no existen. Ejecute mlops_pipeline.py primero.")

        model = joblib.load(model_path)
        vectorizer = joblib.load(vec_path)
        
        # Inicialización de analizadores externos (PySentimiento/VADER) para comparación 
        analyzer_es = create_analyzer(task="sentiment", lang="es") 
        analyzer_en = SentimentIntensityAnalyzer()
        
        return model, vectorizer, analyzer_es, analyzer_en
    except Exception as e:
        raise RuntimeError(f"Falla al cargar recursos de evaluación: {e}")

def run_benchmarking():
    """Ejecuta la comparación de modelos y extrae las características principales."""
    try:
        print("--- Iniciando Etapa: Extracción de caracteristicas principales ---")
        model, vectorizer, analyzer_es, analyzer_en = load_resources()
        
        res = []
        for lang, f in [('ES', 'processed_spanish.csv'), ('EN', 'processed_english.csv')]:
            path = DATA_DIR / f
            if path.exists():
                print(f"Procesando comparación para idioma: {lang}")
                df = pd.read_csv(path)
                
                # Validación de input
                if df.empty:
                    continue
                
                # Preparar textos para nuestro modelo Regresión Logisitica (Pros + Cons)
                texts_for_model = df['pros_clean'].fillna('') + " " + df['cons_clean'].fillna('')
                preds_nuestro = model.predict(vectorizer.transform(texts_for_model))
                
                # Iteración para comparar con modelos (pysentimiento/vader) 
                for i, row in df.iterrows():
                    # Lógica de predicción (Simplificada a binario: 1=Pos/Neu, 0=Neg)
                    if lang == 'ES':
                        # Pysentimiento: POS/NEU se consideran positivos (1) en este contexto
                        ext_pred = 1 if analyzer_es.predict(str(row['pros'])).output in ['POS', 'NEU'] else 0
                    else:
                        # VADER: Compound >= 0 se considera positivo (1)
                        ext_pred = 1 if analyzer_en.polarity_scores(str(row['pros']))['compound'] >= 0 else 0
                    
                    # Se agregan 'pros' y 'cons' al diccionario
                    res.append({
                        'pros': row['pros'],
                        'cons': row['cons'],
                        'idioma': lang,
                        'pred_logReg': preds_nuestro[i],
                        'pred_PyS_VAD': ext_pred
                    })

        if not res:
            raise ValueError("No se generaron datos. Verifique que existan archivos procesados en 'datos/clean_data'.")
            
        # Generación del DataFrame final con todas las características
        df_final = pd.DataFrame(res)
        output_path = BASE_DIR / "datos"  / "resultados" / "resultados_finales_comparativos.csv"
        df_final.to_csv(output_path, index=False)
        
        # Métrica de coincidencia para el log
        match_rate = (df_final['pred_logReg'] == df_final['pred_PyS_VAD']).mean() * 100
        print(f"\n¡Éxito! Comparación finalizada.")
        print(f"Archivo generado: {output_path}")
        print(f"Nivel de coincidencia: {match_rate:.2f}%")
        
    except Exception as e:
        print(f"Error crítico en la Etapa Extracción de caracteristicas principales: {e}")

if __name__ == "__main__":
    run_benchmarking()