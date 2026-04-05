# Glassdoor Sentiment Analysis: NLP & MLOps Challenge
## Master's in Data Science (MCD) - Universidad de Guadalajara
**Author:** Jesús Antonio Ramos

This project implements an end-to-end Machine Learning pipeline to extract, preprocess, and classify Glassdoor reviews from Target employees, comparing a custom Logistic Regression model against industry standards (pySentiment and VADER) using MLOps best practices.

---

### 1. Web Scraping / Dataset Extraction
The extraction pipeline (`scraper_pipeline.py`) utilizes **Selenium** and **Undetected Chromedriver** to bypass anti-bot protections on Glassdoor.
* **Source:** Target Corp reviews page.
* **Methodology:** Navigates through multiple pages (max 15), allowing for secure login and data harvesting.
* **Output:** Generates a raw dataset at `datos/raw_data/raw_target_data.csv`.

### 2. Model Construction (Architecture)
The NLP core converts raw text into a predictive classification tool using a mathematical approach:
* **Preprocessing:** Language detection (langdetect), lemmatization via **SpaCy**, and N-gram distribution analysis (2nd order).
* **Classification:** **Logistic Classifier** trained on TF-IDF vectorized text (max 50 features).
* **Mathematical Foundation:** The model computes **Logistic Sigmoid Probabilities** to distinguish between "Pros" and "Cons".
* **Metadata:** Upon training, the system generates `classification_metadata.txt` with details on architecture and sample size.

### 3. MLOps Integration (MLflow)
Lifecycle management is handled by **MLflow** in `mlops_pipeline.py`, ensuring transparency and reproducibility:
* **Tracking:** Automated logging of metrics (Accuracy, Precision, Recall) and hyperparameters (`max_features`, `ngram_range`).
* **Artifacts:** Storage of the Confusion Matrix and serialized `.pkl` models.
* **Versioning:** Localhost server management for experiment auditing and model registry.

---

### 4. Technical Instructions

#### **Prerequisites**
* Python 3.10+
* Google Chrome (Version 146)
* MLflow Server (`pip install mlflow`)

#### **Execution Flow**
1. **Scraping:** `python src/scraper_pipeline.py`
2. **Preprocessing:** `python src/preprocessing.py` (Generates clean datasets and N-gram stats).
3. **MLflow UI:** Open a new terminal and run `mlflow ui --port 5000`.
4. **Training & MLOps:** `python src/mlops_pipeline.py` (Registers the model in the server).
5. **Evaluation:** `python src/evaluation.py` (Benchmarks against PySentimiento and VADER).

---

### 5. Extra: Post a Request (Inference)
Once the model is served via MLflow, you can classify new reviews by sending a POST request:

**Using Curl:**
```bash
curl -X POST -H "Content-Type:application/json" \
     --data '{"dataframe_split": {"columns": ["text"], "data": [["Great work environment and benefits"]]}}' \
     [http://127.0.0.1:5000/invocations](http://127.0.0.1:5000/invocations)