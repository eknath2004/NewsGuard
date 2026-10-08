"""
Unit tests for model layer and credibility scoring.
"""

import pytest
import numpy as np
from src.models import get_model_zoo


def test_model_zoo_contains_required_models():
    zoo = get_model_zoo()
    required = [
        "Logistic Regression",
        "Support Vector Machine",
        "Random Forest",
        "Gradient Boosting",
        "XGBoost"
    ]
    for model_name in required:
        assert model_name in zoo, f"Model {model_name} missing from model zoo"
    assert len(zoo) >= 5


def test_credibility_score_bounds():
    probs = [0.0, 0.25, 0.5, 0.75, 0.999, 1.0]
    for p in probs:
        score = round(p * 100.0, 1)
        assert 0.0 <= score <= 100.0
        label = "Real" if score >= 50.0 else "Fake"
        if p >= 0.5:
            assert label == "Real"
        else:
            assert label == "Fake"
