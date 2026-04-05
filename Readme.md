# Glassdoor Sentiment Analysis: NLP & MLOps Challenge  
## Master's in Data Science (MCD) - Universidad de Guadalajara

This project implements an end-to-end Machine Learning pipeline to extract, preprocess, and classify Glassdoor reviews from Target employees, comparing the implemented model (Logistic Regression) against industry standards (pySentiment and VADER) through MLOps practices.

### 1. **Web Scraping / Dataset extraction**  
The extraction pipeline (`scraper_pipeline.py`) utilizes **Selenium** and **Undetected Chromedriver** to bypass anti-bot protections on Glassdoor.
* **Source:** Target Corp reviews page.
* **Methodology:** Navigates through multiple pages (max 15), allowing for secure login and data harvesting.
* **Output:** Generates a raw dataset at `datos/raw_data/raw_target_data.csv`.

### 2. **Model Construction**  
The NLP core converts raw text into a predictive classification tool using a mathematical approach:
* **Preprocessing:** Language detection (langdetect), lemmatization via **SpaCy**, and N-gram distribution analysis (2nd order).
* **Classification:** **Logistic Classifier** trained on TF-IDF vectorized text (max 50 features).
* **Mathematical Foundation:** The model computes **Logistic Sigmoid Probabilities** to distinguish between "Pros" and "Cons".
* **Metadata:** Upon training, the system generates `classification_metadata.txt` with details on architecture and sample size.

### 3. **MLOps Integration**  
Lifecycle management is handled by **MLflow** in `mlops_pipeline.py`, ensuring transparency and reproducibility:
* **Tracking:** Automated logging of metrics (Accuracy, Precision, Recall) and hyperparameters (`max_features`, `ngram_range`).
* **Artifacts:** Storage of the Confusion Matrix and serialized `.pkl` models.
* **Versioning:** Localhost server management for experiment auditing and model registry. 

---

### 4. **Technical Instructions**

#### **Prerequisites**
* Python 3.10+
* Google Chrome (Version 146)
* MLflow Server (`pip install mlflow`)

**Installation**
1. Clone the repository:  
git clone https://github.com/jesusantonio5742/proyecto_challenge_basico.git  
cd proyecto_challenge_basico  
2. Install dependencies:  
pip install -r requirements.txt  
python -m spacy download es_core_news_sm  
python -m spacy download en_core_web_sm  

**Execution Flow in terminal**  

1. **Scraping:** `python src/scraper_pipeline.py`
2. **Preprocessing:** `python src/preprocessing.py` (Generates clean datasets and N-gram stats).
3. **MLflow UI:** Open a new terminal and run `mlflow ui --port 5000`.
4. **Training & MLOps:** `python src/mlops_pipeline.py` (Registers the model in the server).
5. **Evaluation:** `python src/evaluation.py` (Benchmarks against PySentimiento and VADER).

The folder named "datos" will contain child folders with the following files:    

    Proyecto_challenge_basico  
    │──datos  
    │     ├──raw_data: the results from the scraper_pipeline.py will be located here  
    │     ├──clean_data: the results from the preprocessing.py will be located here  
    │     ├──modelos: the results from the mlops_pipeline.py will be located here  
    │     ├──resultados: the results from evaluation.py will be located here  

#### **Deployment & Git Workflow**
To sync this local project with the official repository, follow these commands:
```bash
git init
git add .
git commit -m "Final submission: Glassdoor Challenge"
git branch -M main
git remote add origin [https://github.com/jesusantonio5742/proyecto_challenge_basico.git](https://github.com/jesusantonio5742/proyecto_challenge_basico.git)
git pull origin main --rebase
git push -u origin main