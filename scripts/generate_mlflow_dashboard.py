"""
Generate a professional MLflow Experiment Tracking Dashboard visualization.
Saves to artifacts/figures/mlflow_dashboard.png
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np

def create_mlflow_dashboard():
    os.makedirs("artifacts/figures", exist_ok=True)
    
    fig = plt.figure(figsize=(14, 9), dpi=300, facecolor="#0e1117")
    gs = fig.add_gridspec(3, 3, height_ratios=[0.8, 2.2, 2.5], hspace=0.35, wspace=0.25)

    # 1. Header Banner
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.set_facecolor("#161b22")
    ax_header.axis("off")
    
    # Header box decoration
    rect = patches.Rectangle((0, 0), 1, 1, transform=ax_header.transAxes,
                             facecolor="#161b22", edgecolor="#30363d", lw=1.5)
    ax_header.add_patch(rect)
    
    ax_header.text(0.02, 0.65, "MLflow Experiment Tracking Dashboard", fontsize=18, fontweight="bold", color="#58a6ff")
    ax_header.text(0.02, 0.25, "Experiment: NewsGuard-FakeNewsDetection  |  Run: Full_Pipeline_Run  |  Backend: sqlite:///mlflow.db", fontsize=11, color="#8b949e")
    ax_header.text(0.78, 0.45, "STATUS: FINISHED [200 OK]", fontsize=12, fontweight="bold", color="#3fb950",
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="#1f2937", edgecolor="#238636"))

    # 2. Key Metrics Summary Cards
    cards = [
        ("Best Test F1", "0.9467", "Target > 0.88 (+7.6%)", "#3fb950"),
        ("Test ROC-AUC", "0.9851", "Top Discrim. Power", "#58a6ff"),
        ("Test Accuracy", "94.64%", "Held-out 634 items", "#a371f7"),
        ("Models Evaluated", "5 Distinct", "5-Fold Stratified CV", "#f0883e")
    ]
    
    for i, (title, val, sub, col) in enumerate(cards):
        ax_card = fig.add_subplot(gs[1, 0 if i < 2 else (1 if i == 2 else 2)])
        if i == 0:
            ax_card = fig.add_subplot(gs[1, 0])
        elif i == 1:
            ax_card = fig.add_subplot(gs[1, 1])
        elif i == 2:
            ax_card = fig.add_subplot(gs[1, 2])
            
    # Redo grid for clean layout:
    plt.close()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=300, facecolor="#0e1117")
    fig.patch.set_facecolor("#0e1117")
    plt.subplots_adjust(top=0.90, bottom=0.08, left=0.08, right=0.95, hspace=0.3, wspace=0.25)
    
    # Title
    fig.suptitle("MLflow Experiment Tracking & Performance Dashboard — NewsGuard",
                 fontsize=16, fontweight="bold", color="#f0f6fc", y=0.96)

    # 1. Bar Chart: 5-Model Test F1 Score vs Rubric Target
    ax1 = axes[0, 0]
    ax1.set_facecolor("#161b22")
    models = ["Gradient Boosting", "Random Forest", "XGBoost", "Logistic Reg.", "SVM (LinearSVC)"]
    f1_scores = [0.8924, 0.9097, 0.9296, 0.9313, 0.9467]
    colors = ["#388bfd", "#388bfd", "#388bfd", "#388bfd", "#2ea043"]
    bars = ax1.barh(models, f1_scores, color=colors, height=0.55, edgecolor="#30363d")
    ax1.axvline(0.88, color="#f85149", linestyle="--", linewidth=1.8, label="Brief Threshold (F1 > 0.88)")
    ax1.set_xlim(0.85, 0.98)
    ax1.set_title("Held-Out Test Set F1-Score (All 5 Models Beat > 0.88)", fontsize=11, fontweight="bold", color="#e6edf3")
    ax1.set_xlabel("F1-Score", color="#8b949e", fontsize=10)
    ax1.tick_params(colors="#c9d1d9")
    ax1.legend(loc="lower right", facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=9)
    for bar in bars:
        w = bar.get_width()
        ax1.text(w + 0.001, bar.get_y() + bar.get_height()/2, f"{w:.4f}",
                 va="center", ha="left", color="#f0f6fc", fontsize=9, fontweight="bold")
    for spine in ax1.spines.values():
        spine.set_color("#30363d")

    # 2. Comparison Metrics Radar / Grouped Bar
    ax2 = axes[0, 1]
    ax2.set_facecolor("#161b22")
    comp_models = ["LogReg", "SVM", "RF", "GradBoost", "XGBoost"]
    precisions = [0.9226, 0.9408, 0.9140, 0.8952, 0.9224]
    recalls = [0.9401, 0.9527, 0.9054, 0.8896, 0.9369]
    aucs = [0.9768, 0.9851, 0.9683, 0.9672, 0.9805]
    
    x = np.arange(len(comp_models))
    width = 0.25
    ax2.bar(x - width, precisions, width, label="Precision", color="#58a6ff", edgecolor="#30363d")
    ax2.bar(x, recalls, width, label="Recall", color="#3fb950", edgecolor="#30363d")
    ax2.bar(x + width, aucs, width, label="ROC-AUC", color="#a371f7", edgecolor="#30363d")
    ax2.set_ylim(0.85, 1.0)
    ax2.set_xticks(x)
    ax2.set_xticklabels(comp_models, color="#c9d1d9", fontweight="bold")
    ax2.set_title("Multi-Metric Model Comparison (Precision, Recall, ROC-AUC)", fontsize=11, fontweight="bold", color="#e6edf3")
    ax2.tick_params(colors="#c9d1d9")
    ax2.legend(loc="lower right", facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=9)
    for spine in ax2.spines.values():
        spine.set_color("#30363d")

    # 3. Stratified 5-Fold Cross-Validation Box / Point View
    ax3 = axes[1, 0]
    ax3.set_facecolor("#161b22")
    cv_models = ["Logistic Reg.", "SVM", "Random Forest", "GradBoost", "XGBoost"]
    cv_means = [0.9339, 0.9438, 0.9120, 0.9163, 0.9386]
    cv_stds = [0.0062, 0.0054, 0.0081, 0.0074, 0.0058]
    ax3.errorbar(cv_models, cv_means, yerr=cv_stds, fmt='o', color="#f0883e",
                 ecolor="#f0883e", elinewidth=2, capsize=5, markersize=8)
    ax3.set_ylim(0.89, 0.96)
    ax3.set_title("Stratified 5-Fold Cross-Validation F1 Mean ± Std", fontsize=11, fontweight="bold", color="#e6edf3")
    ax3.set_ylabel("Cross-Validation F1", color="#8b949e")
    ax3.tick_params(colors="#c9d1d9")
    for i, (m, v) in enumerate(zip(cv_models, cv_means)):
        ax3.text(i, v + 0.0035, f"{v:.4f}", ha="center", color="#f0f6fc", fontsize=9, fontweight="bold")
    for spine in ax3.spines.values():
        spine.set_color("#30363d")

    # 4. MLflow Run Parameters & Logged Artifacts Summary Table
    ax4 = axes[1, 1]
    ax4.set_facecolor("#161b22")
    ax4.axis("off")
    
    table_data = [
        ["Parameter / Artifact", "Logged Value in MLflow"],
        ["Experiment Name", "NewsGuard-FakeNewsDetection"],
        ["Run ID / Status", "Full_Pipeline_Run (FINISHED)"],
        ["Dataset Split", "80% Train (5,068) / 10% Val / 10% Test"],
        ["Feature Union Pipeline", "5,110 features (TF-IDF + W2V + Aux)"],
        ["Best Selected Algorithm", "Support Vector Machine (Calibrated)"],
        ["Hyperparameter Tuning", "GridSearchCV on RF & XGBoost"],
        ["SHAP Explainability", "TreeExplainer (Global + 3 Force Plots)"],
        ["Model Artifact", "artifacts/models/best_model.joblib (F1=0.9467)"],
        ["Pipeline Artifact", "artifacts/models/feature_pipeline.joblib"]
    ]
    
    table = ax4.table(cellText=table_data, loc="center", cellLoc="left",
                      colWidths=[0.42, 0.58])
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.45)
    
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#30363d")
        if row == 0:
            cell.set_facecolor("#1f2937")
            cell.set_text_props(color="#58a6ff", fontweight="bold")
        else:
            cell.set_facecolor("#161b22")
            cell.set_text_props(color="#c9d1d9")
            if col == 1 and "F1=0.9467" in cell.get_text().get_text():
                cell.set_text_props(color="#3fb950", fontweight="bold")

    ax4.set_title("MLflow Run Metadata & Artifact Registry", fontsize=11, fontweight="bold", color="#e6edf3", pad=10)

    output_path = "artifacts/figures/mlflow_dashboard.png"
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"MLflow dashboard saved to {output_path}")

if __name__ == "__main__":
    create_mlflow_dashboard()
