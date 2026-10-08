"""
Train and export best tree model (XGBoost) for SHAP TreeExplainer in the serving layer.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import joblib
import pandas as pd
from xgboost import XGBClassifier

from src.preprocessor import clean_raw_text

def export_tree_model():
    print("Loading train dataset and feature pipeline...")
    train_df = pd.read_csv("data/processed/train.csv")
    pipeline = joblib.load("artifacts/models/feature_pipeline.joblib")
    
    raw_texts = train_df["content"].fillna("").tolist()
    clean_texts = [clean_raw_text(t) for t in raw_texts]
    y_train = train_df["label"].values


    print("Transforming training features...")
    X_train = pipeline.transform(clean_texts)

    print("Fitting XGBoost classifier for SHAP TreeExplainer...")
    tree_model = XGBClassifier(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42
    )
    tree_model.fit(X_train, y_train)

    out_path = "artifacts/models/tree_model.joblib"
    joblib.dump(tree_model, out_path)
    print(f"Saved tree model to {out_path}")

if __name__ == "__main__":
    export_tree_model()
