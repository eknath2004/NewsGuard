"""
Flask REST API for NewsGuard:
- POST /predict: Predicts real/fake label, 0-100 credibility score, top SHAP features
- Proper error handling: Returns HTTP 400 for malformed input
- GET /health: Healthcheck endpoint
- GET /metrics: Returns 5-model comparison metrics
- CORS enabled
"""

import os
import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

from src.pipeline import NewsGuardPredictor

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("NewsGuardAPI")

app = Flask(__name__)
CORS(app)

# Global predictor instance
_PREDICTOR = None


def get_predictor():
    global _PREDICTOR
    if _PREDICTOR is None:
        model_path = os.path.abspath("artifacts/models/best_model.joblib")
        pipeline_path = os.path.abspath("artifacts/models/feature_pipeline.joblib")
        if os.path.exists(model_path) and os.path.exists(pipeline_path):
            logger.info("Loading NewsGuardPredictor...")
            _PREDICTOR = NewsGuardPredictor(
                model_path=model_path,
                feature_pipeline_path=pipeline_path
            )
        else:
            logger.warning("Model artifacts not yet generated. Please run pipeline first.")
    return _PREDICTOR


@app.route("/", methods=["GET"])
def index():
    """Root endpoint with API overview."""
    return jsonify({
        "system": "NewsGuard — Fake News Detection & Credibility Scoring API",
        "version": "1.0.0",
        "endpoints": {
            "GET /health": "Health check",
            "POST /predict": "Predict news credibility and SHAP linguistic indicators",
            "GET /metrics": "5-model benchmark evaluation table"
        },
        "docs": "Send POST request to /predict with JSON: {'title': '...', 'text': '...'}"
    }), 200


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    predictor = get_predictor()
    model_loaded = predictor is not None and predictor.model is not None
    return jsonify({
        "status": "healthy",
        "model_loaded": model_loaded,
        "version": "1.0.0"
    }), 200


@app.route("/metrics", methods=["GET"])
def metrics():
    """Return model comparison benchmark metrics."""
    metrics_path = "artifacts/metrics/5_model_comparison.csv"
    meta_path = "artifacts/models/model_metadata.json"

    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    if os.path.exists(metrics_path):
        import pandas as pd
        df = pd.read_csv(metrics_path)
        return jsonify({
            "status": "success",
            "metadata": metadata,
            "models_comparison": df.to_dict(orient="records")
        }), 200
    else:
        return jsonify({
            "status": "pending",
            "message": "Benchmark metrics not yet computed."
        }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Expects JSON body:
    {
        "title": "Optional Article Title",
        "text": "Full article text to evaluate..."
    }
    Returns real/fake label, 0-100 credibility score, linguistic metrics, and top SHAP features.
    Malformed inputs return HTTP 400.
    """
    # 1. Validate Content-Type and JSON validity
    if not request.is_json:
        return jsonify({
            "error": "Malformed input: Request Content-Type must be 'application/json'",
            "status_code": 400
        }), 400

    try:
        data = request.get_json(silent=True)
    except Exception:
        data = None

    if data is None or not isinstance(data, dict):
        return jsonify({
            "error": "Malformed input: Request body must be valid JSON object.",
            "status_code": 400
        }), 400

    # 2. Extract and validate text fields
    title = str(data.get("title", "")).strip()
    text = str(data.get("text", "")).strip()

    if not text:
        return jsonify({
            "error": "Malformed input: 'text' field is required and cannot be empty.",
            "status_code": 400
        }), 400

    if len(text) < 10 and len(title) < 5:
        return jsonify({
            "error": "Malformed input: News article text is too short to evaluate (minimum 10 characters required).",
            "status_code": 400
        }), 400

    # 3. Perform Inference
    predictor = get_predictor()
    if predictor is None:
        return jsonify({
            "error": "Model is currently initializing or not trained yet. Please train pipeline first.",
            "status_code": 503
        }), 503

    try:
        result = predictor.predict(text=text, title=title)
        return jsonify({
            "status": "success",
            **result
        }), 200
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        return jsonify({
            "error": f"Internal prediction error: {str(e)}",
            "status_code": 500
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting NewsGuard REST API server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
