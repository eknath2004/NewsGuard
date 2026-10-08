"""
Model Evaluation and Selection Layer:
- Evaluates 5+ classifiers on held-out test set
- Generates 5-Model Comparison Table (Precision, Recall, F1, AUC)
- Plots and saves Confusion Matrices and ROC Curves
- Selects and exports best model via joblib with F1 > 0.88 verification
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix,
    classification_report
)


def evaluate_and_compare_models(
    models: dict,
    X_train,
    y_train: np.ndarray,
    X_test,
    y_test: np.ndarray,
    figures_dir: str = "artifacts/figures",
    metrics_dir: str = "artifacts/metrics",
    models_dir: str = "artifacts/models"
) -> tuple[pd.DataFrame, str, any]:
    """
    Fits each model on X_train, evaluates on held-out test set,
    generates comparison artifacts, and exports best model.
    """
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    comparison_rows = []
    fitted_models = {}
    test_predictions = {}
    test_probabilities = {}

    print(f"\n=======================================================")
    print(f"  TRAINING & EVALUATING 5 CLASSIFICATION ALGORITHMS")
    print(f"=======================================================\n")

    for name, model in models.items():
        print(f"--> Training {name} on {X_train.shape[0]} training samples...")
        model.fit(X_train, y_train)
        fitted_models[name] = model

        # Predictions and Probabilities
        y_pred = model.predict(X_test)
        test_predictions[name] = y_pred

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            df_vals = model.decision_function(X_test)
            # Sigmoid normalization for models with decision_function
            y_prob = 1.0 / (1.0 + np.exp(-df_vals))
        else:
            y_prob = y_pred.astype(float)
        test_probabilities[name] = y_prob

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        comparison_rows.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1_Score": round(f1, 4),
            "ROC_AUC": round(auc, 4)
        })

    # Convert to DataFrame and sort by F1_Score descending
    df_comparison = pd.DataFrame(comparison_rows).sort_values(by="F1_Score", ascending=False).reset_index(drop=True)

    # Save CSV and Markdown
    csv_path = os.path.join(metrics_dir, "5_model_comparison.csv")
    md_path = os.path.join(metrics_dir, "5_model_comparison.md")
    df_comparison.to_csv(csv_path, index=False)
    try:
        md_table = df_comparison.to_markdown(index=False)
    except Exception:
        md_table = df_comparison.to_string(index=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 5-Model Comparison Benchmark Table\n\n")
        f.write(md_table)

    print("\n[Held-Out Test Set 5-Model Comparison Table]")
    print(df_comparison.to_string(index=False))

    # Plot Confusion Matrices
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 3.8), dpi=300)
    if n_models == 1:
        axes = [axes]

    for idx, (name, _) in enumerate(models.items()):
        cm = confusion_matrix(y_test, test_predictions[name])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=axes[idx],
            xticklabels=["Fake", "Real"],
            yticklabels=["Fake", "Real"]
        )
        axes[idx].set_title(f"{name}\n(F1: {df_comparison[df_comparison['Model']==name]['F1_Score'].values[0]:.3f})", fontsize=10, fontweight="bold")
        axes[idx].set_xlabel("Predicted Label", fontsize=9)
        if idx == 0:
            axes[idx].set_ylabel("True Label", fontsize=9)
        else:
            axes[idx].set_ylabel("")

    plt.tight_layout()
    cm_fig_path = os.path.join(figures_dir, "confusion_matrices.png")
    plt.savefig(cm_fig_path)
    plt.close()
    print(f"[Evaluation] Saved confusion matrices to {cm_fig_path}")

    # Plot ROC Curves
    plt.figure(figsize=(8, 6), dpi=300)
    palette = ["#2980b9", "#27ae60", "#e67e22", "#8e44ad", "#e74c3c", "#16a085"]
    for idx, (name, _) in enumerate(models.items()):
        fpr, tpr, _ = roc_curve(y_test, test_probabilities[name])
        auc_val = df_comparison[df_comparison["Model"] == name]["ROC_AUC"].values[0]
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", color=palette[idx % len(palette)], lw=2)

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.title("ROC Curves Comparison Across 5 Classification Algorithms", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True, facecolor="white")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_fig_path = os.path.join(figures_dir, "roc_curves.png")
    plt.savefig(roc_fig_path)
    plt.close()
    print(f"[Evaluation] Saved ROC curves to {roc_fig_path}")

    # Identify Best Model
    best_row = df_comparison.iloc[0]
    best_name = best_row["Model"]
    best_f1 = float(best_row["F1_Score"])
    best_model = fitted_models[best_name]

    print(f"\n=======================================================")
    print(f"  BEST MODEL SELECTED: {best_name}")
    print(f"  Test F1-Score: {best_f1:.4f} (Target: > 0.88)")
    print(f"=======================================================\n")

    # Save detailed classification report for best model
    best_report = classification_report(y_test, test_predictions[best_name], target_names=["Fake", "Real"], output_dict=True)
    report_path = os.path.join(metrics_dir, "best_model_classification_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(best_report, f, indent=2)

    # Export best model via joblib
    export_path = os.path.join(models_dir, "best_model.joblib")
    joblib.dump(best_model, export_path)
    print(f"[Evaluation] Exported best model to {export_path}")

    # Save model metadata
    meta = {
        "best_model_name": best_name,
        "f1_score": best_f1,
        "accuracy": float(best_row["Accuracy"]),
        "precision": float(best_row["Precision"]),
        "recall": float(best_row["Recall"]),
        "roc_auc": float(best_row["ROC_AUC"]),
        "meets_rubric_f1_target": bool(best_f1 > 0.88),
        "exported_path": export_path
    }
    with open(os.path.join(models_dir, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return df_comparison, best_name, best_model


if __name__ == "__main__":
    print("Model evaluation module ready.")
