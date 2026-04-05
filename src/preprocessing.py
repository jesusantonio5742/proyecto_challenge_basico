import pandas as pd
import re
from pathlib import Path
from langdetect import detect, DetectorFactory
import spacy
from collections import Counter

DetectorFactory.seed = 0
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "datos" / "raw_data" / "raw_target_data.csv"
OUTPUT_DIR = BASE_DIR / "datos" / "clean_data"

# Se carga modelo spaCy
def load_spacy_models():
    try:
        return spacy.load("es_core_news_sm"), spacy.load("en_core_web_sm")
    except Exception as e:
        raise RuntimeError(f"Error al cargar modelos de lenguaje: {e}")

def save_ngrams_with_probability(series, n, filename):
    try:
        all_words = " ".join(series.dropna().astype(str)).split()
        if not all_words: return
        n_grams_list = list(zip(*[all_words[i:] for i in range(n)]))
        total_ngrams = len(n_grams_list)
        gram_counts = Counter(n_grams_list)
        df_counts = pd.DataFrame(gram_counts.most_common(20), columns=['N-gram', 'Frecuencia'])
        df_counts['Probabilidad (%)'] = (df_counts['Frecuencia'] / total_ngrams * 100).round(4)
        df_counts['N-gram'] = df_counts['N-gram'].apply(lambda x: "-".join(x))
        df_counts.to_csv(OUTPUT_DIR / filename, index=False)
    except Exception as e:
        print(f"Error guardando n-gramas {filename}: {e}")

def run_preprocessing_pipeline():
    try:
        print("Iniciando Etapa Preprocessing...")
        if not INPUT_FILE.exists():
            raise FileNotFoundError(f"No se encontró el input del scraper en {INPUT_FILE}")

        # Validación datos de entrada
        df = pd.read_csv(INPUT_FILE)
        if df.empty or 'pros' not in df.columns:
            raise ValueError("El archivo de entrada está corrupto o vacío.")

        nlp_es, nlp_en = load_spacy_models()
        df['temp_full'] = df['pros'].fillna('') + " " + df['cons'].fillna('')
        df['lang'] = df['temp_full'].apply(lambda x: detect(x) if len(x) > 5 else "unknown")
        
        for lang, name, nlp in [('es', 'spanish', nlp_es), ('en', 'english', nlp_en)]:
            sub_df = df[df['lang'] == lang].copy()
            if not sub_df.empty:
                for col in ['pros', 'cons']:
                    sub_df[f'{col}_clean'] = sub_df[col].apply(lambda x: " ".join([t.lemma_ for t in nlp(re.sub(r'[^a-zñáéíóúü\s]', '', str(x).lower())) if not t.is_stop]))
                
                save_ngrams_with_probability(sub_df['pros_clean'], 2, f"stats_{name}_2gram.csv")
                sub_df.to_csv(OUTPUT_DIR / f"processed_{name}.csv", index=False)
                
        print("Preprocessing finalizado con validaciones exitosas.")
    except Exception as e:
        print(f"Error crítico en Preprocessing: {e}")

if __name__ == "__main__":
    run_preprocessing_pipeline()