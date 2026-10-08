# 🛡️ NewsGuard — Fake News Detection & Credibility Scoring System

[![Python 3.12](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Model F1](https://img.shields.io/badge/Held--Out%20Test%20F1-0.9467%20(%3E0.88)-success.svg)](artifacts/metrics/5_model_comparison.md)
[![Tests Passing](https://img.shields.io/badge/Tests-17%2F17%20Passed-brightgreen.svg)](tests/)
[![MLflow](https://img.shields.io/badge/Experiment%20Tracking-MLflow-0194E2.svg)](mlflow.db)

> **Code-A-Nova Internship Program 2026 — Machine Learning Track | Task 5**  
> An enterprise-grade, end-to-end NLP machine learning pipeline that classifies news articles as **Real** or **Fake**, derives a calibrated **0–100 Credibility Score**, visualizes linguistic and SHAP explainability indicators, and serves real-time predictions via a **Flask REST API** and an interactive **Gradio Web Interface**.

---

## 📑 Table of Contents
1. [Architecture & Data Flow](#-architecture--data-flow)
2. [Tech Stack](#-tech-stack)
3. [Key Highlights & Benchmark Results](#-key-highlights--benchmark-results)
4. [Project Structure](#-project-structure)
5. [10-Day Milestone Implementation](#-10-day-milestone-implementation)
6. [Installation & Setup](#-installation--setup)
7. [Running the Pipeline](#-running-the-pipeline)
8. [Flask REST API Documentation](#-flask-rest-api-documentation)
9. [Interactive Gradio Web Interface](#-interactive-gradio-web-interface)
10. [Explainability (SHAP TreeExplainer)](#-explainability-shap-treeexplainer)
11. [Testing Suite](#-testing-suite)
12. [Model Card](#-model-card)

---

## 🏛️ Architecture & Data Flow

```mermaid
flowchart TD
    subgraph Data_Layer ["1. Data Layer & Ingestion"]
        A[Raw Corpus: 6,335 News Articles] --> B[Stratified Split: 80% Train / 10% Val / 10% Test]
        B --> C[EDA: Class Distribution & Word Lengths]
    end

    subgraph Preprocessing_Layer ["2. Text Preprocessing Pipeline (No Data Leakage)"]
        B -->|Train Only| D[URL, HTML, Noise & Email Stripping]
        D --> E[SpaCy Lemmatization & Stopword Pruning]
    end

    subgraph Feature_Union ["3. Scikit-Learn FeatureUnion (5,110 Total Features)"]
        E --> F[TF-IDF Vectorizer: 1-gram + 2-gram, 5,000 Vocab]
        E --> G[Gensim Word2Vec: 100-dim Skip-Gram Sentence Vectors]
        D --> H[Auxiliary Linguistic Extractor: Readability, Sentiment, Style]
        F & G & H --> I[Combined Feature Matrix: 5,110 Dense/Sparse Features]
    end

    subgraph Model_Zoo ["4. Multi-Model Benchmark (Stratified 5-Fold CV)"]
        I --> J1[Logistic Regression]
        I --> J2[Calibrated SVM / LinearSVC]
        I --> J3[Random Forest]
        I --> J4[Gradient Boosting]
        I --> J5[XGBoost Classifier]
        J1 & J2 & J3 & J4 & J5 --> K[GridSearchCV Hyperparameter Tuning]
        K --> L["Model Selection: Best Model (F1 = 0.9467 > 0.88)"]
    end

    subgraph Explainability_Layer ["5. SHAP TreeExplainer & Credibility Engine"]
        L --> M[Calibrated Softmax Probability P(Real)]
        M --> N["Credibility Score = round(P(Real) * 100, 1)"]
        L --> O[SHAP TreeExplainer Global Summary Plot]
        L --> P[SHAP Force Plots: Real, Fake, Borderline Case Studies]
    end

    subgraph Serving_Layer ["6. Deployment & Serving"]
        N & P --> Q["Flask REST API (POST /predict, GET /metrics, GET /health)"]
        N & P --> R["Gradio Web UI (Visual Meter Gauge & Dynamic SHAP Bar Plot)"]
    end
```

---

## 🛠️ Tech Stack

| Category | Technology | Purpose in NewsGuard |
| :--- | :--- | :--- |
| **Language** | Python 3.12+ | Core programming runtime |
| **NLP Preprocessing** | SpaCy (`en_core_web_sm`), NLTK | Lemmatization, POS tags, stopword removal |
| **Statistical Features** | Scikit-Learn `TfidfVectorizer` | Unigram + bigram n-gram vocabulary (5,000 max features) |
| **Embedding Features** | Gensim `Word2Vec` | 100-dim Skip-Gram document vector embeddings |
| **Auxiliary Features** | TextStat, TextBlob | Flesch-Kincaid readability, reading ease, sentiment polarity & subjectivity |
| **ML Algorithms (5 Distinct)** | Scikit-Learn, XGBoost | Logistic Regression, SVM (`LinearSVC`), Random Forest, Gradient Boosting, XGBoost |
| **Model Tuning** | Scikit-Learn `GridSearchCV` | Hyperparameter optimization on top tree models |
| **Model Evaluation** | Scikit-Learn Metrics | Stratified 5-fold CV, Confusion Matrix, ROC-AUC curves |
| **Explainability** | SHAP (`TreeExplainer`) | Global feature attributions, individual force plots, top-N word metrics |
| **REST API** | Flask, Flask-CORS | Production endpoints with input validation and HTTP 400 error handlers |
| **Interactive UI** | Gradio | Web application with visual credibility gauge, SHAP chart, sample articles |
| **Experiment Tracking** | MLflow | SQLite-backed run logging for parameters, metrics, and model artifacts |
| **Testing** | Pytest | 17 unit and integration tests covering all pipeline stages |

---

## 🏆 Key Highlights & Benchmark Results

### 1. Held-Out Test Set Performance (634 Unseen Articles)
> **Rubric Requirement:** Achieve F1 > 0.88 on held-out test set with 5 distinct classification algorithms.

| Classification Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Status vs Target (>0.88) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **Support Vector Machine (LinearSVC)** | **94.64%** | **0.9408** | **0.9527** | **0.9467** | **0.9851** | **Exceeds (+7.58%)** |
| 🥈 **Logistic Regression** | 93.06% | 0.9226 | 0.9401 | 0.9313 | 0.9768 | **Exceeds (+5.83%)** |
| 🥉 **XGBoost Classifier** | 92.90% | 0.9224 | 0.9369 | 0.9296 | 0.9805 | **Exceeds (+5.64%)** |
| 🏅 **Random Forest** | 91.01% | 0.9140 | 0.9054 | 0.9097 | 0.9683 | **Exceeds (+3.38%)** |
| 🏅 **Gradient Boosting** | 89.27% | 0.8952 | 0.8896 | 0.8924 | 0.9672 | **Exceeds (+1.41%)** |

*All 5 classification algorithms comfortably beat the F1 > 0.88 project benchmark!*

### 2. Stratified 5-Fold Cross-Validation (5,068 Training Articles)
- **Support Vector Machine:** Mean CV F1 = **0.9438**, Mean CV AUC = **0.9861**
- **XGBoost Classifier:** Mean CV F1 = **0.9386**, Mean CV AUC = **0.9833**
- **Logistic Regression:** Mean CV F1 = **0.9339**, Mean CV AUC = **0.9816**
- **Gradient Boosting:** Mean CV F1 = **0.9163**, Mean CV AUC = **0.9727**
- **Random Forest:** Mean CV F1 = **0.9120**, Mean CV AUC = **0.9737**

### 3. Derived Credibility Score
The credibility score is strictly derived from the calibrated posterior probability distribution:
$$\text{Credibility Score} = \text{round}\left(P(\text{Class} = \text{Real} \mid \vec{x}) \times 100, 1\right)$$
- **80 – 100:** Highly Credible (Green)
- **60 – 79:** Likely Credible (Teal)
- **40 – 59:** Uncertain / Borderline (Yellow)
- **20 – 39:** Suspicious (Orange)
- **0 – 19:** Fabricated / Fake (Red)

---

## 📂 Project Structure

```
assing 3/
├── api/
│   └── app.py                      # Flask REST API (POST /predict, GET /metrics, GET /health)
├── ui/
│   └── gradio_app.py               # Interactive Gradio UI with visual credibility gauge
├── src/
│   ├── data_loader.py              # Ingestion, EDA figures, stratified 80/10/10 splitting
│   ├── preprocessor.py             # SpaCy lemmatization, noise removal, Scikit-learn transformer
│   ├── feature_extraction.py       # TF-IDF, Word2VecEmbedder, AuxiliaryLinguisticExtractor, FeatureUnion
│   ├── models.py                   # 5 model zoo definitions, 5-fold CV, GridSearchCV tuning
│   ├── explainability.py           # SHAP TreeExplainer, summary plot, force plot generation
│   └── pipeline.py                 # Full orchestrator & production NewsGuardPredictor class
├── scripts/
│   ├── generate_mlflow_dashboard.py # MLflow Experiment Tracking dashboard visualizer
│   └── export_tree_explainer.py    # Standalone tree-explainer model exporter
├── tests/
│   ├── test_api.py                 # API tests (200, 400 bad JSON, empty text, short text)
│   ├── test_features.py            # FeatureUnion, Word2Vec, Auxiliary tests
│   ├── test_model.py               # 5-model zoo & credibility bounds tests
│   └── test_preprocessor.py        # SpaCy lemmatizer, URL/HTML/email stripper tests
├── artifacts/
│   ├── figures/                    # High-resolution figures (EDA, ROC, Confusion Matrices, MLflow)
│   ├── explainability/             # SHAP summary plot, HTML & PNG force plots (Real, Fake, Borderline)
│   ├── metrics/                    # 5_model_comparison.csv, cv_comparison.csv, json metrics
│   └── models/                     # best_model.joblib, feature_pipeline.joblib, word2vec.model
├── data/
│   ├── raw/                        # Ingested benchmark datasets (6,335 articles + LIAR corpus)
│   └── processed/                  # Stratified train.csv, val.csv, test.csv
├── MODEL_CARD.md                   # Documented Model Card (Deliverable 9)
├── README.md                       # Comprehensive Project Documentation (Deliverable 10)
├── requirements.txt                # Production dependencies
└── mlflow.db                       # MLflow SQLite experiment tracking database
```

---

## 📅 10-Day Milestone Implementation

| Day | Milestone Goal | Implementation Status | Artifact Deliverable |
| :---: | :--- | :---: | :--- |
| **Day 1** | Dataset setup, class distribution analysis, stratified 80/10/10 split, EDA | ✅ Complete | `data/processed/{train,val,test}.csv`, `eda_class_distribution.png`, `eda_length_distribution.png` |
| **Day 2** | Text preprocessing pipeline — tokenization, stopword removal, SpaCy lemmatization, URL/noise stripping | ✅ Complete | `src/preprocessor.py`, `TextPreprocessor` transformer, unit tests |
| **Day 3** | TF-IDF features — unigram + bigram vectorizer, vocabulary analysis, discriminative terms visualization | ✅ Complete | `artifacts/figures/discriminative_terms.png`, `artifacts/metrics/discriminative_terms.json` |
| **Day 4** | Auxiliary features — Word2Vec sentence vectors, readability (Flesch-Kincaid), sentiment, length | ✅ Complete | `Word2VecEmbedder` (Gensim), `AuxiliaryLinguisticExtractor` (TextStat + TextBlob) |
| **Day 5** | Scikit-Learn `FeatureUnion` pipeline; baseline models (Logistic Regression), stratified 5-fold CV | ✅ Complete | `artifacts/models/feature_pipeline.joblib` (5,110 features), CV comparison table |
| **Day 6** | Advanced models — SVM, Random Forest, Gradient Boosting, XGBoost; hyperparameter tuning | ✅ Complete | `GridSearchCV` on RF (`max_depth=35, n_estimators=150`) and XGBoost (`max_depth=5, n_estimators=120`) |
| **Day 7** | Model evaluation — confusion matrix, Precision/Recall/F1, ROC-AUC, export best model | ✅ Complete | `artifacts/figures/confusion_matrices.png`, `roc_curves.png`, `best_model.joblib` (F1 = 0.9467) |
| **Day 8** | SHAP explainability — TreeExplainer, force plots for individual predictions, summary plot | ✅ Complete | `shap_summary.png`, `force_plot_{real,fake,borderline}.png/.html` |
| **Day 9** | Flask API (`POST /predict`), Gradio UI with article input and visual credibility gauge | ✅ Complete | `api/app.py`, `ui/gradio_app.py` |
| **Day 10** | Model card documentation, MLflow experiment summary, test suite, README | ✅ Complete | `MODEL_CARD.md`, `artifacts/figures/mlflow_dashboard.png`, 17 passed unit tests |

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.11 or Python 3.12
- Git

### 1. Clone & Set Up Environment
```bash
git clone <repository-url>
cd "assing 3"

# Create virtual environment (optional but recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download required SpaCy & NLTK models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"
```

---

## ⚡ Running the Pipeline

To re-train the models from scratch, perform 5-fold cross-validation, run hyperparameter tuning, compute SHAP attributions, and log everything to MLflow:

```bash
python -m src.pipeline
```

The pipeline will:
1. Load and stratify the 6,335-sample dataset (80/10/10).
2. Clean text and fit the `FeatureUnion` (5,110 features).
3. Execute stratified 5-fold cross-validation across all 5 algorithms.
4. Tune hyperparameters via `GridSearchCV` on Random Forest and XGBoost.
5. Evaluate all 5 models on the unseen 634-sample test set.
6. Verify F1 > 0.88 and export `artifacts/models/best_model.joblib`.
7. Generate SHAP global summary plot and 3 force plot case studies.

---

## 🌐 Flask REST API Documentation

### Starting the Flask Server
```bash
python -m api.app
# Server listens on http://localhost:5000
```

### Endpoints Overview

| Method | Endpoint | Description | Status Code |
| :---: | :--- | :--- | :---: |
| `GET` | `/` | API status and overview | 200 OK |
| `GET` | `/health` | Healthcheck and model readiness probe | 200 OK |
| `GET` | `/metrics` | 5-model evaluation benchmark table | 200 OK |
| `POST` | `/predict` | Predict authenticity, credibility score & SHAP features | 200 OK / 400 Bad Request |

### `POST /predict` Request & Response Specification

#### Request Payload (`application/json`):
```json
{
  "title": "International Delegates Sign Clean Energy Cooperation Framework",
  "text": "GENEVA — Diplomatic representatives from forty-two nations concluded multilateral trade and environmental negotiations on Thursday, agreeing to standardized cross-border energy frameworks and transparency protocols."
}
```

#### Success Response (`200 OK`):
```json
{
  "status": "success",
  "label": "Real",
  "credibility_score": 98.6,
  "verdict": "Highly Credible",
  "badge_color": "green",
  "confidence": 0.986,
  "probabilities": {
    "real": 0.986,
    "fake": 0.014
  },
  "linguistic_metrics": {
    "flesch_kincaid_grade": 12.8,
    "flesch_reading_ease": 42.1,
    "sentiment_polarity": 0.05,
    "sentiment_subjectivity": 0.22,
    "word_count": 27,
    "exclamation_ratio": 0.0,
    "uppercase_ratio": 0.04
  },
  "top_features": [
    {
      "feature": "Term: 'contributed report'",
      "raw_name": "tfidf__contributed report",
      "shap_value": 0.5326,
      "feature_value": 0.0,
      "direction": "Supports Real",
      "importance_percentage": 0.5326
    },
    {
      "feature": "Linguistic Metric: aux_uppercase_ratio",
      "raw_name": "aux__aux_uppercase_ratio",
      "shap_value": 0.4282,
      "feature_value": -0.4356,
      "direction": "Supports Real",
      "importance_percentage": 0.4282
    }
  ],
  "summary": "NewsGuard evaluated this content as 'Real' with a Credibility Score of 98.6/100 (Highly Credible). Sentiment polarity is 0.05 and reading grade level is 12.8."
}
```

#### Error Handling (`400 Bad Request`):
Malformed requests (missing body, empty string, non-JSON `Content-Type`, or text < 10 characters) return standard HTTP 400:
```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": ""}'
```
Response:
```json
{
  "error": "Malformed input: 'text' field is required and cannot be empty.",
  "status_code": 400
}
```

---

## 💻 Interactive Gradio Web Interface

### Starting the Gradio App
```bash
python -m ui.gradio_app
# Interface runs locally at: http://localhost:7860
```

### Key UI Features:
1. **Headline & Article Input Box:** Accommodates headlines and full-body text.
2. **Visual Credibility Meter Gauge:** Dynamic HTML/CSS meter with color-coded badges (Green, Teal, Yellow, Orange, Red).
3. **SHAP Word Importance Chart:** Real-time horizontal bar plot displaying the strongest positive (green) and negative (red) linguistic indicators.
4. **Linguistic Metrics Card:** Displays Flesch-Kincaid Grade Level, Sentiment Polarity, Subjectivity, and Stylometric Ratios.
5. **1-Click Benchmark Examples:** Preloaded sample articles (Diplomatic Accord, Sensationalist Alien Conspiracy, Political Budget Analysis) for instant testing.

---

## 🔍 Explainability (SHAP TreeExplainer)

NewsGuard features interpretable AI through `shap.TreeExplainer`:
- **Global Feature Importance (`artifacts/explainability/shap_summary.png`):** Shows the global ranking and impact direction across hundreds of unseen articles.
- **Individual Prediction Force Plots (`artifacts/explainability/`):**
  - `force_plot_real.png` / `.html`: High credibility sample pushed by institutional terminology and objective sentiment.
  - `force_plot_fake.png` / `.html`: Low credibility sample pulled down by sensationalism, high exclamation frequency, and emotional polarity.
  - `force_plot_borderline.png` / `.html`: Ambiguous sample with conflicting statistical indicators.

---

## 🧪 Testing Suite

The test suite validates data preprocessing, feature union pipelines, model zoos, score bounds, and REST API error handling.

Run all tests:
```bash
python -m pytest tests/ -v
```

### Test Coverage Summary:
- `tests/test_api.py`: Root endpoint (200), Health check (200), Metrics endpoint (200), Valid prediction (200), Malformed non-JSON (400), Empty text (400), Short text (400).
- `tests/test_features.py`: `Word2VecEmbedder` output shape, `AuxiliaryLinguisticExtractor` metric verification, `FeatureUnion` dimension checks.
- `tests/test_model.py`: 5 distinct model zoo validation, 0–100 credibility score bounds and label mappings.
- `tests/test_preprocessor.py`: Regex URL cleaning, HTML tag removal, email stripping, SpaCy lemmatization, stopword pruning.

**Result: 17 Passed in 20s (100% Passing)**

---

## 📋 Model Card

Detailed documentation regarding model architecture, training distribution, evaluation metrics, quantitative performance tables, limitations, and ethical considerations is available in [MODEL_CARD.md](MODEL_CARD.md).

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
