"""
Unit tests for text preprocessing pipeline.
"""

import pytest
import pandas as pd
from src.preprocessor import clean_raw_text, lemmatize_and_filter, TextPreprocessor


def test_clean_raw_text_urls():
    text = "Visit https://news.example.com/breaking and http://test.org for updates."
    cleaned = clean_raw_text(text)
    assert "https://" not in cleaned
    assert "http://" not in cleaned
    assert "updates" in cleaned


def test_clean_raw_text_html():
    text = "<h1>Breaking Alert</h1><p>Major development reported.</p>"
    cleaned = clean_raw_text(text)
    assert "<h1>" not in cleaned
    assert "<p>" not in cleaned
    assert "Breaking Alert Major development reported." in cleaned


def test_clean_raw_text_emails():
    text = "For press inquiries contact media@reuters.com or editor@news.org immediately."
    cleaned = clean_raw_text(text)
    assert "media@reuters.com" not in cleaned
    assert "inquiries contact" in cleaned


def test_lemmatize_and_filter():
    text = "Scientists are investigating several mysterious occurrences across various European nations."
    lemmas = lemmatize_and_filter(text)
    assert isinstance(lemmas, str)
    assert len(lemmas.split()) > 0
    # Common stopwords like 'are', 'across' should be removed
    lemma_tokens = lemmas.split()
    assert "are" not in lemma_tokens


def test_text_preprocessor_transformer():
    texts = [
        "Economy ministers announced new quarterly trade forecasts today.",
        "Shocking hidden truth revealed by online blogger!!"
    ]
    transformer = TextPreprocessor(batch_size=10)
    transformed = transformer.fit_transform(texts)
    assert len(transformed) == 2
    assert isinstance(transformed[0], str)
    assert len(transformed[0]) > 0
