"""
Text Preprocessing Pipeline:
- Tokenization, stopword removal, SpaCy lemmatization
- URL, email, HTML, noise removal, and punctuation stripping
- Scikit-Learn compatible transformer (TextPreprocessor)
"""

import re
import string
import spacy
from typing import List, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.corpus import stopwords as nltk_stopwords

# Pre-load stopwords
try:
    NLTK_STOPS = set(nltk_stopwords.words("english"))
except Exception:
    NLTK_STOPS = set()

# Cache SpaCy model
_NLP = None

def get_spacy_nlp():
    """
    Load SpaCy en_core_web_sm model with parser and ner disabled
    for ultra-fast lemmatization.
    """
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        except Exception:
            _NLP = spacy.blank("en")
    return _NLP


def clean_raw_text(text: str) -> str:
    """
    Remove URLs, HTML tags, emails, non-ASCII artifacts, and excess whitespace.
    Preserves case and punctuation info for linguistic feature extractors before lemmatization.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)
    # Remove email addresses
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", " ", text)
    # Remove twitter handles/mentions
    text = re.sub(r"@\w+", " ", text)
    # Normalize unicode whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def lemmatize_and_filter(text: str, nlp=None, stop_words: set = None) -> str:
    """
    Tokenize, remove stopwords, strip punctuation, and lemmatize using SpaCy.
    """
    if nlp is None:
        nlp = get_spacy_nlp()
    if stop_words is None:
        stop_words = NLTK_STOPS

    cleaned = clean_raw_text(text)
    if not cleaned:
        return ""

    doc = nlp(cleaned.lower())
    lemmas = []
    for token in doc:
        # Filter stopwords, punctuation, digits, symbols, and single-char non-words
        if (
            not token.is_punct
            and not token.is_space
            and not token.is_stop
            and token.lemma_ not in stop_words
            and len(token.lemma_) > 2
            and token.is_alpha
        ):
            lemmas.append(token.lemma_)

    return " ".join(lemmas)


def batch_preprocess_texts(
    texts: Union[List[str], pd.Series, np.ndarray],
    batch_size: int = 256,
    n_process: int = 1
) -> List[str]:
    """
    Batch process texts efficiently using nlp.pipe for high throughput.
    """
    nlp = get_spacy_nlp()
    stop_words = NLTK_STOPS

    # Pre-clean raw text
    cleaned_texts = [clean_raw_text(t) for t in texts]

    processed = []
    for doc in nlp.pipe(cleaned_texts, batch_size=batch_size, n_process=n_process):
        lemmas = [
            token.lemma_.lower()
            for token in doc
            if (
                not token.is_punct
                and not token.is_space
                and not token.is_stop
                and token.lemma_.lower() not in stop_words
                and len(token.lemma_) > 2
                and token.is_alpha
            )
        ]
        processed.append(" ".join(lemmas))

    return processed


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-Learn Transformer for text preprocessing.
    Integrates seamlessly into Scikit-Learn pipelines with zero data leakage.
    """

    def __init__(self, batch_size: int = 256):
        self.batch_size = batch_size

    def fit(self, X, y=None):
        # Text preprocessing is stateless, no fitting required
        return self

    def transform(self, X):
        """
        Transforms input text array or series into lemmatized text list.
        """
        if isinstance(X, pd.Series):
            text_list = X.fillna("").tolist()
        elif isinstance(X, np.ndarray):
            text_list = [str(x) for x in X.ravel()]
        elif isinstance(X, list):
            text_list = [str(x) for x in X]
        else:
            text_list = [str(X)]

        return batch_preprocess_texts(text_list, batch_size=self.batch_size)


if __name__ == "__main__":
    sample_text = (
        "Breaking News! Scientists at https://stanford.edu discovered that climate policies "
        "have improved air quality dramatically by 2026. Contact info@news.org for details."
    )
    print("Raw text:\n", sample_text)
    clean = clean_raw_text(sample_text)
    print("\nCleaned text:\n", clean)
    lemmatized = lemmatize_and_filter(sample_text)
    print("\nLemmatized tokens:\n", lemmatized)
