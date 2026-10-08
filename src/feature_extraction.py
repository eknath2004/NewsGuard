"""
Feature Engineering Pipeline:
- TF-IDF Vectorizer (unigram + bigram)
- Word2Vec Sentence/Document Embeddings (Gensim)
- Auxiliary Linguistic Features (Readability, Sentiment, Article Length, Punctuation)
- Scikit-Learn FeatureUnion for Sparse + Dense feature combination
- Vocabulary & Discriminative Terms Analysis
"""

import os
import re
import json
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.feature_selection import chi2

from gensim.models import Word2Vec
import textstat
from textblob import TextBlob


class Word2VecEmbedder(BaseEstimator, TransformerMixin):
    """
    Transforms text documents into average Word2Vec embedding vectors.
    Trained strictly on the training set to prevent data leakage.
    Returns scipy csr_matrix for seamless FeatureUnion integration.
    """

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        workers: int = 4,
        epochs: int = 10,
        sg: int = 1,
        model_path: str = "artifacts/models/word2vec.model"
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.sg = sg
        self.model_path = model_path
        self.model = None

    def _tokenize(self, text: str) -> list[str]:
        return [w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", str(text))]

    def fit(self, X, y=None):
        sentences = [self._tokenize(text) for text in X]
        # If corpus is small and no tokens satisfy min_count, dynamically adjust min_count to 1
        effective_min_count = self.min_count
        word_counts = {}
        for s in sentences:
            for w in s:
                word_counts[w] = word_counts.get(w, 0) + 1
        if not any(c >= effective_min_count for c in word_counts.values()):
            effective_min_count = 1

        if not word_counts:
            sentences = [["news", "article", "report"]]
            effective_min_count = 1

        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=effective_min_count,
            workers=self.workers,
            sg=self.sg,
            epochs=self.epochs,
            seed=42
        )
        # Ensure target directory exists and save model
        if self.model_path:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.model.save(self.model_path)
        return self

    def transform(self, X):
        if self.model is None:
            if self.model_path and os.path.exists(self.model_path):
                self.model = Word2Vec.load(self.model_path)
            else:
                raise ValueError("Word2Vec model is not fitted yet.")

        wv = self.model.wv
        embeddings = np.zeros((len(X), self.vector_size), dtype=np.float32)

        for i, text in enumerate(X):
            tokens = self._tokenize(text)
            vectors = [wv[w] for w in tokens if w in wv]
            if vectors:
                embeddings[i] = np.mean(vectors, axis=0)

        # Return sparse matrix to ensure FeatureUnion compatibility
        return sparse.csr_matrix(embeddings)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        return [f"w2v_dim_{i}" for i in range(self.vector_size)]


class AuxiliaryLinguisticExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts readability, sentiment, article length, and punctuation statistics.
    Scaled using StandardScaler fitted strictly on training data.
    Returns scipy csr_matrix for seamless FeatureUnion integration.
    """

    FEATURE_NAMES = [
        "flesch_reading_ease",
        "flesch_kincaid_grade",
        "sentiment_polarity",
        "sentiment_subjectivity",
        "char_count",
        "word_count",
        "avg_word_length",
        "exclamation_ratio",
        "question_ratio",
        "uppercase_ratio"
    ]

    def __init__(self):
        self.scaler = StandardScaler()

    def _extract_single(self, text: str) -> list[float]:
        raw_text = str(text) if text is not None else ""
        char_count = len(raw_text)
        words = raw_text.split()
        word_count = len(words)
        avg_word_len = char_count / max(word_count, 1)

        # Punctuation & casing ratios
        exclamation_count = raw_text.count("!")
        question_count = raw_text.count("?")
        upper_count = sum(1 for c in raw_text if c.isupper())

        exclamation_ratio = exclamation_count / max(char_count, 1)
        question_ratio = question_count / max(char_count, 1)
        uppercase_ratio = upper_count / max(char_count, 1)

        # Fast slice for linguistic and readability metrics
        slice_text = raw_text[:500]

        # Readability metrics
        try:
            reading_ease = float(textstat.flesch_reading_ease(slice_text))
            reading_grade = float(textstat.flesch_kincaid_grade(slice_text))
        except Exception:
            reading_ease = 50.0
            reading_grade = 8.0

        # Sentiment metrics via TextBlob
        try:
            blob = TextBlob(slice_text)
            polarity = float(blob.sentiment.polarity)
            subjectivity = float(blob.sentiment.subjectivity)
        except Exception:
            polarity = 0.0
            subjectivity = 0.5

        return [
            reading_ease,
            reading_grade,
            polarity,
            subjectivity,
            float(char_count),
            float(word_count),
            avg_word_len,
            exclamation_ratio,
            question_ratio,
            uppercase_ratio
        ]

    def fit(self, X, y=None):
        features = [self._extract_single(text) for text in X]
        self.scaler.fit(features)
        return self

    def transform(self, X):
        features = [self._extract_single(text) for text in X]
        scaled_features = self.scaler.transform(features)
        return sparse.csr_matrix(scaled_features)

    def fit_transform(self, X, y=None):
        features = [self._extract_single(text) for text in X]
        scaled_features = self.scaler.fit_transform(features)
        return sparse.csr_matrix(scaled_features)

    def extract_raw_metrics(self, text: str) -> dict[str, float]:
        """Utility for API and UI to get human-readable unscaled metrics."""
        raw = self._extract_single(text)
        return dict(zip(self.FEATURE_NAMES, [round(val, 4) for val in raw]))

    def get_feature_names_out(self, input_features=None):
        return [f"aux_{name}" for name in self.FEATURE_NAMES]


def build_feature_union_pipeline(
    max_tfidf_features: int = 5000,
    w2v_dim: int = 100,
    w2v_model_path: str = "artifacts/models/word2vec.model"
) -> FeatureUnion:
    """
    Construct Scikit-Learn FeatureUnion combining:
    1. Sparse TF-IDF (unigram + bigram)
    2. Dense Word2Vec sentence embeddings (as CSR sparse)
    3. Dense Auxiliary linguistic & readability features (as CSR sparse)
    """
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_tfidf_features,
        sublinear_tf=True,
        stop_words="english"
    )

    w2v = Word2VecEmbedder(
        vector_size=w2v_dim,
        model_path=w2v_model_path
    )

    aux = AuxiliaryLinguisticExtractor()

    feature_union = FeatureUnion(
        transformer_list=[
            ("tfidf", tfidf),
            ("w2v", w2v),
            ("aux", aux)
        ]
    )
    return feature_union


def analyze_discriminative_terms(
    tfidf_vectorizer: TfidfVectorizer,
    X_train_texts: list[str],
    y_train: np.ndarray,
    top_k: int = 15,
    figures_dir: str = "artifacts/figures",
    metrics_dir: str = "artifacts/metrics"
) -> dict:
    """
    Perform chi-squared feature selection to identify the most discriminative terms
    for Real vs Fake news. Saves visualization and metrics JSON.
    """
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    X_tfidf = tfidf_vectorizer.transform(X_train_texts)
    chi2_scores, p_values = chi2(X_tfidf, y_train)
    feature_names = np.array(tfidf_vectorizer.get_feature_names_out())

    # Get class mean TF-IDF to determine direction
    real_mask = (y_train == 1)
    fake_mask = (y_train == 0)

    real_mean = np.asarray(X_tfidf[real_mask].mean(axis=0)).ravel()
    fake_mean = np.asarray(X_tfidf[fake_mask].mean(axis=0)).ravel()
    diff = real_mean - fake_mean  # Positive -> supports Real, Negative -> supports Fake

    # Real indicators (high chi2, positive diff)
    real_indices = np.where(diff > 0)[0]
    top_real_idx = real_indices[np.argsort(chi2_scores[real_indices])[-top_k:][::-1]]

    # Fake indicators (high chi2, negative diff)
    fake_indices = np.where(diff < 0)[0]
    top_fake_idx = fake_indices[np.argsort(chi2_scores[fake_indices])[-top_k:][::-1]]

    discriminative_data = {
        "top_real_terms": [
            {"term": feature_names[i], "chi2": float(chi2_scores[i]), "diff": float(diff[i])}
            for i in top_real_idx
        ],
        "top_fake_terms": [
            {"term": feature_names[i], "chi2": float(chi2_scores[i]), "diff": float(diff[i])}
            for i in top_fake_idx
        ]
    }

    # Save JSON
    with open(os.path.join(metrics_dir, "discriminative_terms.json"), "w", encoding="utf-8") as f:
        json.dump(discriminative_data, f, indent=2)

    # Plot top discriminative terms
    plt.figure(figsize=(12, 6), dpi=300)
    
    # Subplot 1: Real terms
    plt.subplot(1, 2, 1)
    real_terms = [item["term"] for item in discriminative_data["top_real_terms"]]
    real_scores = [item["chi2"] for item in discriminative_data["top_real_terms"]]
    sns.barplot(x=real_scores, y=real_terms, color="#2ecc71")
    plt.title("Top Terms Indicative of Real News", fontsize=12, fontweight="bold")
    plt.xlabel("Chi-Squared Score", fontsize=10)

    # Subplot 2: Fake terms
    plt.subplot(1, 2, 2)
    fake_terms = [item["term"] for item in discriminative_data["top_fake_terms"]]
    fake_scores = [item["chi2"] for item in discriminative_data["top_fake_terms"]]
    sns.barplot(x=fake_scores, y=fake_terms, color="#e74c3c")
    plt.title("Top Terms Indicative of Fake News", fontsize=12, fontweight="bold")
    plt.xlabel("Chi-Squared Score", fontsize=10)

    plt.tight_layout()
    fig_path = os.path.join(figures_dir, "discriminative_terms.png")
    plt.savefig(fig_path)
    plt.close()

    print(f"[Feature Extraction] Saved discriminative terms visualization to {fig_path}")
    return discriminative_data


if __name__ == "__main__":
    from src.data_loader import load_news_dataset, perform_stratified_split
    df = load_news_dataset()
    train_df, _, _ = perform_stratified_split(df)
    
    sample_texts = train_df["content"].head(50).tolist()
    pipeline = build_feature_union_pipeline(max_tfidf_features=1000, w2v_dim=50)
    print("Fitting FeatureUnion on sample texts...")
    features = pipeline.fit_transform(sample_texts)
    print(f"Combined Feature Matrix Shape: {features.shape} (Sparse format: {sparse.issparse(features)})")
