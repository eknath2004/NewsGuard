"""
Tests for Flask REST API endpoints and error handling.
"""

import pytest
import json
from api.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_root_endpoint(client):
    res = client.get("/")
    assert res.status_code == 200
    data = res.get_json()
    assert "NewsGuard" in data["system"]
    assert "POST /predict" in data["endpoints"]


def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "healthy"


def test_predict_malformed_not_json(client):
    # Missing application/json Content-Type
    res = client.post("/predict", data="plain text data")
    assert res.status_code == 400
    data = res.get_json()
    assert "error" in data


def test_predict_empty_text(client):
    # Empty text payload
    res = client.post(
        "/predict",
        data=json.dumps({"title": "Test Title", "text": ""}),
        content_type="application/json"
    )
    assert res.status_code == 400
    data = res.get_json()
    assert "error" in data


def test_predict_too_short(client):
    res = client.post(
        "/predict",
        data=json.dumps({"text": "abc"}),
        content_type="application/json"
    )
    assert res.status_code == 400
    data = res.get_json()
    assert "error" in data


def test_metrics_endpoint(client):
    res = client.get("/metrics")
    assert res.status_code == 200
    data = res.get_json()
    assert "status" in data


def test_predict_valid_article(client):
    sample_text = (
        "The Secretary of State delivered a keynote address at the international "
        "diplomatic symposium, outlining multilateral trade agreements and treaty frameworks "
        "signed by representative ambassadors from fifteen member nations."
    )
    res = client.post(
        "/predict",
        data=json.dumps({"title": "International Trade Framework Announced", "text": sample_text}),
        content_type="application/json"
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert data["label"] in ["Real", "Fake"]
    assert 0.0 <= data["credibility_score"] <= 100.0
    assert "probabilities" in data
    assert "top_features" in data
    assert isinstance(data["top_features"], list)
    assert "linguistic_metrics" in data

