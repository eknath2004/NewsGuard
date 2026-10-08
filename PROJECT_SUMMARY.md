# 🎓 Code-A-Nova Internship Program 2026 — Task 5 Final Project Report

**Project Title:** NewsGuard — Fake News Detection & Credibility Scoring System  
**Track:** Machine Learning Engineering  
**Approved By:** HR / Lead ML Mentor  
**Date of Completion:** October 2026  
**Status:** ✅ 100% Complete & Verified  

---

## Executive Summary

The **NewsGuard** project delivers an end-to-end Machine Learning and Natural Language Processing solution for classifying news articles, computing a calibrated credibility score (0–100), explaining model predictions with SHAP feature attributions, and serving predictions via a Flask REST API and an interactive Gradio web application.

The best model (**Calibrated Support Vector Machine / LinearSVC**) achieved an **F1-Score of 0.9467** on an unseen held-out test set (634 articles), exceeding the rubric requirement of **F1 > 0.88** by **+7.58%**. All 5 evaluated models surpassed the threshold.

---

## 10 Deliverables Verification Checklist

| # | Deliverable Specification (from Brief) | File Location / Evidence | Status |
| :-: | :--- | :--- | :-: |
| **1** | **Preprocessed dataset with train/val/test split (stratified)** | `data/processed/train.csv` (5,068)<br>`data/processed/val.csv` (633)<br>`data/processed/test.csv` (634) | ✅ Verified (80/10/10 Stratified) |
| **2** | **Feature engineering pipeline (`FeatureUnion`: TF-IDF + Word2Vec + Auxiliary)** | `artifacts/models/feature_pipeline.joblib`<br>`artifacts/models/word2vec.model`<br>5,110 total features | ✅ Verified |
| **3** | **5-model comparison table: Precision, Recall, F1, AUC per model** | `artifacts/metrics/5_model_comparison.csv`<br>`artifacts/metrics/5_model_comparison.md` | ✅ Verified |
| **4** | **Best model exported via `joblib` with F1 > 0.88 on held-out test set** | `artifacts/models/best_model.joblib`<br>**Test F1: 0.9467** (AUC: 0.9851) | ✅ Verified (Exceeds > 0.88) |
| **5** | **SHAP summary plot & 3 force plot examples for individual predictions** | `artifacts/explainability/shap_summary.png`<br>`force_plot_real.{png,html}`<br>`force_plot_fake.{png,html}`<br>`force_plot_borderline.{png,html}` | ✅ Verified (Interactive & PNG) |
| **6** | **Flask API (`POST /predict`) returning label, credibility score, and top SHAP features** | `api/app.py`<br>Endpoints: `/predict`, `/health`, `/metrics`<br>Proper HTTP 400 on malformed input | ✅ Verified & Tested |
| **7** | **Gradio or Streamlit UI with article input & visual credibility gauge** | `ui/gradio_app.py`<br>Features: Credibility meter, SHAP bar plot, linguistic cards, 1-click test examples | ✅ Verified & Tested |
| **8** | **MLflow experiment tracking dashboard screenshot / visual summary** | `artifacts/figures/mlflow_dashboard.png`<br>`mlflow.db` (SQLite tracking store) | ✅ Verified |
| **9** | **Model Card document (training data, metrics, limitations, intended use)** | `MODEL_CARD.md` | ✅ Verified |
| **10** | **pytest test suite & README with setup and API documentation** | `tests/` (17/17 tests passing)<br>`README.md` (Full architecture, curl docs, setup) | ✅ Verified |

---

## Benchmark Results Summary

### Stratified 5-Fold Cross-Validation (5,068 Samples)
- **Support Vector Machine:** Precision = 0.9444, Recall = 0.9432, **F1 = 0.9438**, AUC = 0.9861
- **XGBoost Classifier:** Precision = 0.9403, Recall = 0.9369, **F1 = 0.9386**, AUC = 0.9833
- **Logistic Regression:** Precision = 0.9314, Recall = 0.9365, **F1 = 0.9339**, AUC = 0.9816
- **Gradient Boosting:** Precision = 0.9156, Recall = 0.9172, **F1 = 0.9163**, AUC = 0.9727
- **Random Forest:** Precision = 0.9234, Recall = 0.9011, **F1 = 0.9120**, AUC = 0.9737

### Held-Out Test Set Performance (634 Unseen Samples)
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Meets Target (>0.88)? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Support Vector Machine (LinearSVC)** | **0.9464** | **0.9408** | **0.9527** | **0.9467** | **0.9851** | **YES** |
| **Logistic Regression** | 0.9306 | 0.9226 | 0.9401 | 0.9313 | 0.9768 | **YES** |
| **XGBoost Classifier** | 0.9290 | 0.9224 | 0.9369 | 0.9296 | 0.9805 | **YES** |
| **Random Forest** | 0.9101 | 0.9140 | 0.9054 | 0.9097 | 0.9683 | **YES** |
| **Gradient Boosting** | 0.8927 | 0.8952 | 0.8896 | 0.8924 | 0.9672 | **YES** |

---

## Architectural Highlights & Technical Solutions
1. **Zero Data Leakage:** All TF-IDF tokenizers, Word2Vec neural embeddings, and auxiliary scalers were fitted strictly on the 80% training set before transforming validation and test partitions.
2. **Dual-Path Explainability:** The system combines high-speed linear feature weights for instant REST API throughput with `shap.TreeExplainer` on the gradient-boosted tree model for deep force plots and global interpretability.
3. **Calibrated Credibility Scoring:** Bounded 0–100 credibility scores are derived directly from the model's calibrated posterior probability ($P(\text{Real}) \times 100$), avoiding subjective ad-hoc formulas.
4. **Resilient Production API:** Flask endpoint validates payload format, JSON headers, and string length, returning descriptive HTTP 400 error responses on invalid data.
