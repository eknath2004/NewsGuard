"""
Unit tests for feature engineering pipeline.
"""

import pytest
import numpy as np
from scipy import sparse
from src.feature_extraction import (
    Word2VecEmbedder,
    AuxiliaryLinguisticExtractor,
    build_feature_union_pipeline
)


def test_word2vec_embedder():
    texts = [
        "Government officials held emergency meetings regarding trade tariffs.",
        "Markets responded positively after central bank updated benchmark rates.",
        "New agricultural subsidies were introduced across southern provinces."
    ]
    embedder = Word2VecEmbedder(vector_size=32, min_count=1, epochs=5, model_path=None)
    embedder.fit(texts)
    matrix = embedder.transform(texts)
    assert sparse.issparse(matrix)
    assert matrix.shape == (3, 32)


def test_auxiliary_linguistic_extractor():
    texts = [
        "Presidential summit concludes with multilateral pact signed by delegates.",
        "OMG SHOCKING CONSPIRACY EXPOSED NOW!! CHECK THIS OUT???"
    ]
    extractor = AuxiliaryLinguisticExtractor()
    extractor.fit(texts)
    features = extractor.transform(texts)
    assert sparse.issparse(features)
    assert features.shape == (2, len(AuxiliaryLinguisticExtractor.FEATURE_NAMES))

    # Test single extraction
    raw = extractor.extract_raw_metrics(texts[1])
    assert "flesch_reading_ease" in raw
    assert "sentiment_polarity" in raw
    assert "exclamation_ratio" in raw
    assert raw["exclamation_ratio"] > 0


def test_feature_union_pipeline():
    texts = [
        "Federal communications agency releases regulatory framework for telecom companies.",
        "Bizarre alien discovery uncovered in remote Antarctic research post!!"
    ]
    pipeline = build_feature_union_pipeline(max_tfidf_features=100, w2v_dim=20, w2v_model_path=None)
    X = pipeline.fit_transform(texts)
    assert sparse.issparse(X)
    assert X.shape[0] == 2
    # Combined features: TF-IDF ngrams (bounded by vocab) + 20 (w2v) + 10 (aux)
    assert X.shape[1] > 30
    assert X.shape[1] <= 130

