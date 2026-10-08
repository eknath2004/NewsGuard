"""
Model Layer:
- 5+ Distinct Classification Algorithms (Logistic Regression, SVM, Random Forest, Gradient Boosting, XGBoost)
- Stratified 5-Fold Cross-Validation Pipeline
- Hyperparameter Tuning with GridSearchCV
"""

import json
import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)


def get_model_zoo(random_state: int = 42) -> Dict[str, Any]:
    """
    Returns dictionary of 5 distinct classification models + baseline Naive Bayes.
    """
    return {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=random_state,
            solver="lbfgs"
        ),
        "Support Vector Machine": CalibratedClassifierCV(
            estimator=LinearSVC(C=1.0, dual=False, max_iter=2000, random_state=random_state),
            cv=3
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=30,
            random_state=random_state,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=60,
            learning_rate=0.1,
            max_depth=3,
            random_state=random_state
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=-1
        )
    }


def evaluate_models_cross_validation(
    models: Dict[str, Any],
    X,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    metrics_dir: str = "artifacts/metrics"
) -> pd.DataFrame:
    """
    Run Stratified 5-Fold Cross-Validation on all models.
    Computes Precision, Recall, F1, Accuracy, and ROC-AUC (mean and std).
    """
    os.makedirs(metrics_dir, exist_ok=True)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    results = []

    print(f"\n=======================================================")
    print(f"  RUNNING STRATIFIED {n_splits}-FOLD CROSS-VALIDATION")
    print(f"=======================================================\n")

    for name, model in models.items():
        print(f"--> Evaluating: {name}...", flush=True)
        cv_res = cross_validate(
            model,
            X,
            y,
            cv=skf,
            scoring=scoring,
            n_jobs=1 if "XGBoost" in name or "Random Forest" in name else -1,
            return_train_score=False
        )

        acc_mean, acc_std = cv_res["test_accuracy"].mean(), cv_res["test_accuracy"].std()
        prec_mean, prec_std = cv_res["test_precision"].mean(), cv_res["test_precision"].std()
        rec_mean, rec_std = cv_res["test_recall"].mean(), cv_res["test_recall"].std()
        f1_mean, f1_std = cv_res["test_f1"].mean(), cv_res["test_f1"].std()
        auc_mean, auc_std = cv_res["test_roc_auc"].mean(), cv_res["test_roc_auc"].std()

        results.append({
            "Model": name,
            "CV_Accuracy_Mean": round(acc_mean, 4),
            "CV_Accuracy_Std": round(acc_std, 4),
            "CV_Precision_Mean": round(prec_mean, 4),
            "CV_Precision_Std": round(prec_std, 4),
            "CV_Recall_Mean": round(rec_mean, 4),
            "CV_Recall_Std": round(rec_std, 4),
            "CV_F1_Mean": round(f1_mean, 4),
            "CV_F1_Std": round(f1_std, 4),
            "CV_AUC_Mean": round(auc_mean, 4),
            "CV_AUC_Std": round(auc_std, 4),
        })

    df_res = pd.DataFrame(results)
    csv_path = os.path.join(metrics_dir, "cv_comparison.csv")
    json_path = os.path.join(metrics_dir, "cv_results.json")
    
    df_res.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n[Cross-Validation Results]")
    print(df_res[["Model", "CV_Precision_Mean", "CV_Recall_Mean", "CV_F1_Mean", "CV_AUC_Mean"]].to_string(index=False))
    return df_res


def tune_hyperparameters(
    model_name: str,
    base_model: Any,
    param_grid: dict,
    X,
    y: np.ndarray,
    cv: int = 3,
    scoring: str = "f1",
    n_jobs: int = -1
) -> Tuple[Any, dict, float]:
    """
    Tune model hyperparameters using GridSearchCV on training data.
    """
    print(f"\n[GridSearchCV] Tuning {model_name}...")
    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=1
    )
    grid.fit(X, y)
    print(f"[GridSearchCV] Best params for {model_name}: {grid.best_params_}")
    print(f"[GridSearchCV] Best CV {scoring}: {grid.best_score_:.4f}")
    return grid.best_estimator_, grid.best_params_, grid.best_score_


if __name__ == "__main__":
    from scipy import sparse
    X_dummy = sparse.csr_matrix(np.random.randn(100, 50))
    y_dummy = np.random.randint(0, 2, size=100)
    zoo = get_model_zoo()
    print("Model Zoo initialized with models:", list(zoo.keys()))
