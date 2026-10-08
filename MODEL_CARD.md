# 📋 Model Card: NewsGuard — Fake News Detection & Credibility Scoring System

## 1. Model Details
- **Model Name:** NewsGuard Credibility Scorer & Fake News Classifier
- **Version:** 1.0.0
- **Model Architecture:** Calibrated Support Vector Machine (`CalibratedClassifierCV` wrapping `LinearSVC(dual=False, C=1.0)`) backed by a multi-modal feature union and secondary Tree-Based Explainer (`XGBoost Classifier`)
- **Developer / Organization:** Code-A-Nova Internship Program 2026 (Task 5: Machine Learning Track)
- **License:** MIT License
- **Frameworks & Libraries:** Python 3.12+, Scikit-Learn 1.8+, XGBoost 3.2+, SHAP 0.51+, SpaCy 3.8+ (`en_core_web_sm`), NLTK 3.9+, Gensim 4.4+, TextBlob 0.19+, TextStat 0.7+, MLflow 3.10+, Flask 3.1+, Gradio 6.9+
- **Artifacts Stored:**
  - Classifier: `artifacts/models/best_model.joblib`
  - Feature Pipeline: `artifacts/models/feature_pipeline.joblib`
  - Word2Vec Embeddings: `artifacts/models/word2vec.model`
  - Tree Explainer Model: `artifacts/models/tree_model.joblib`

---

## 2. Intended Use & Deployment Scope
- **Primary Intended Use:** Automated assessment of English news articles, digital publications, blog posts, and press releases to estimate factual credibility (0–100 score), assign a binary authenticity verdict (Real vs Fake), and generate transparent linguistic/lexical explanations using SHAP feature attribution.
- **Primary Users:** Fact-checkers, newsrooms, content moderation pipelines, journalism students, and digital media researchers.
- **Out-of-Scope & Unintended Uses:**
  - Automated autonomous censorship or removal of articles without editorial/human review.
  - Analyzing non-English texts or languages not supported by the current SpaCy / Word2Vec models.
  - Evaluating creative fiction, satire, parody, poetry, or non-journalistic text domains.
  - Forensic legal or courtroom evidence without independent investigative corroboration.

---

## 3. Training Data & Data Ingestion
- **Dataset Source:** Real vs Fake News Benchmark Corpus (augmented with LIAR dataset ingestion pipelines), comprising 6,335 long-form journalistic documents.
- **Class Balance:**
  - Class 0 (Fake): 3,164 samples (49.94%)
  - Class 1 (Real): 3,171 samples (50.06%)
  - Perfectly balanced distribution ensuring no majority-class prediction bias.
- **Data Partitioning (Zero Data Leakage Protocol):**
  - **Stratified Split:** 80% Training (5,068 samples) / 10% Validation (633 samples) / 10% Held-Out Test (634 samples).
  - All tokenizers, TF-IDF vocabularies, Word2Vec vectorizers, and auxiliary scalers were fitted **exclusively on the training partition** to eliminate data leakage.
- **Preprocessing Pipeline:**
  - Regex stripping of URLs, HTTP protocols, HTML markup tags, and email addresses.
  - SpaCy lemmatization and stopword removal preserving key semantic lemmas.
  - Case folding and punctuation normalization while tracking exclamation and uppercase ratios as auxiliary features.

---

## 4. Feature Engineering (`Scikit-Learn FeatureUnion`)
The model ingests raw text through an integrated `FeatureUnion` yielding **5,110 combined features**:
1. **Statistical N-Gram Features (TF-IDF Vectorizer):**
   - Unigram and bigram representation (`ngram_range=(1, 2)`)
   - Sublinear term frequency scaling (`sublinear_tf=True`)
   - 5,000 maximum vocabulary dimensions filtered by document frequency (`min_df=2`, `max_df=0.95`)
2. **Dense Word Embeddings (Gensim Word2Vec):**
   - 100-dimensional Skip-Gram model (`sg=1`, `window=5`, `epochs=10`)
   - Mean vector pooling over all valid document tokens
3. **Auxiliary Linguistic & Stylometric Features (`AuxiliaryLinguisticExtractor`):**
   - **Readability Scores:** Flesch Reading Ease score, Flesch-Kincaid Grade Level (`textstat`)
   - **Sentiment Analysis:** TextBlob Sentiment Polarity (-1 to +1) and Subjectivity (0 to 1)
   - **Stylistic Metrics:** Total word count, character count, uppercase character ratio, exclamation mark frequency, question mark frequency, and punctuation-to-word density.

---

## 5. Quantitative Evaluation & Benchmark Results

### A. Stratified 5-Fold Cross-Validation (Training Set: 5,068 Samples)
| Classification Model | Mean CV Precision | Mean CV Recall | Mean CV F1-Score | Mean CV ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **Support Vector Machine (LinearSVC)** | **0.9444** | **0.9432** | **0.9438** | **0.9861** |
| **XGBoost Classifier** | 0.9403 | 0.9369 | 0.9386 | 0.9833 |
| **Logistic Regression** | 0.9314 | 0.9365 | 0.9339 | 0.9816 |
| **Gradient Boosting** | 0.9156 | 0.9172 | 0.9163 | 0.9727 |
| **Random Forest** | 0.9234 | 0.9011 | 0.9120 | 0.9737 |

### B. Held-Out Test Set Performance (634 Unseen Articles)
*Project requirement: Achieve F1 > 0.88 on held-out test set.*

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Meets Target (> 0.88)? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Support Vector Machine (Selected)** | **94.64%** | **0.9408** | **0.9527** | **0.9467** | **0.9851** | **YES (+7.58%)** |
| **Logistic Regression** | 93.06% | 0.9226 | 0.9401 | 0.9313 | 0.9768 | **YES (+5.83%)** |
| **XGBoost Classifier** | 92.90% | 0.9224 | 0.9369 | 0.9296 | 0.9805 | **YES (+5.64%)** |
| **Random Forest** | 91.01% | 0.9140 | 0.9054 | 0.9097 | 0.9683 | **YES (+3.38%)** |
| **Gradient Boosting** | 89.27% | 0.8952 | 0.8896 | 0.8924 | 0.9672 | **YES (+1.41%)** |

### C. Detailed Classification Report (Best Model: Support Vector Machine)
```
              precision    recall  f1-score   support

  Fake (0)       0.952     0.940     0.946       317
  Real (1)       0.941     0.953     0.947       317

  accuracy                           0.946       634
 macro avg       0.946     0.946     0.946       634
weighted avg     0.946     0.946     0.946       634
```

---

## 6. Credibility Scoring Formula
Rather than employing arbitrary heuristics, the NewsGuard Credibility Score is mathematically derived from the calibrated posterior probability distribution:
$$\text{Credibility Score} = \text{round}\left(P(\text{Class} = \text{Real} \mid \vec{x}) \times 100, 1\right)$$

- **Score Range:** $[0.0, 100.0]$
- **Verdict Mapping:**
  - $80.0 \le \text{Score} \le 100.0$: **Highly Credible** (Verified Real, Green Badge)
  - $60.0 \le \text{Score} < 80.0$: **Likely Credible** (Substantial Evidence, Teal Badge)
  - $40.0 \le \text{Score} < 60.0$: **Uncertain / Borderline** (Ambiguous Signals, Yellow Badge)
  - $20.0 \le \text{Score} < 40.0$: **Suspicious** (Misinformation Indicators, Orange Badge)
  - $0.0 \le \text{Score} < 20.0$: **Fabricated / Fake** (High Misinformation Risk, Red Badge)

---

## 7. Model Explainability & Interpretability
- **Global Feature Attributions (`artifacts/explainability/shap_summary.png`):**
  - Generated using `shap.TreeExplainer` on held-out test distributions.
  - High positive SHAP values (pushing predictions toward "Real") correlate with objective institutional terms (`reuters`, `spokesman`, `senate`, `minister`, `diplomats`, `statement`), lower emotional polarity, and higher reading grade levels.
  - Negative SHAP values (pushing predictions toward "Fake") correlate with clickbait keywords (`shocking`, `unbelievable`, `conspiracy`, `bombshell`, `illuminati`), excessive exclamation marks (`exclamation_ratio`), and elevated uppercase density (`uppercase_ratio`).
- **Individual Case Studies (`artifacts/explainability/`):**
  - `force_plot_real.png` / `force_plot_real.html`: Case 1 — High Credibility (Real News)
  - `force_plot_fake.png` / `force_plot_fake.html`: Case 2 — Low Credibility (Fake News)
  - `force_plot_borderline.png` / `force_plot_borderline.html`: Case 3 — Borderline News

---

## 8. Limitations & Ethical Considerations
1. **Domain Bias:** The training data is composed of English-language news reporting. Performance may degrade on social media short-text snippets (e.g. 280-character microblogs), tweets, or forum threads without structural body text.
2. **Temporal Drift:** Misinformation tactics evolve rapidly. Emerging slang, new conspiracy theories, and novel world events post-dating training will require periodic retraining and vocabulary refreshes.
3. **Adversarial Perturbations:** Intentional obfuscation (e.g., character swaps, intentional grammatical anomalies, embedding spam) could potentially alter TF-IDF feature distributions.
4. **Fairness & Objectivity:** The model evaluates linguistic and stylistic cues, not objective real-time truth verification databases. It should serve as an augmentation tool for human analysts.
