"""
Explainability Layer:
- SHAP TreeExplainer on tree-based models (Random Forest, XGBoost)
- Summary Plot Generation
- 3 Individual Force Plot / Waterfall Examples (Real, Fake, Borderline)
- Top-N Feature Word Importance Extraction per prediction
"""

import os
import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import sparse


class NewsGuardExplainer:
    """
    SHAP-based explainability module for NewsGuard.
    Provides global feature importance summary plots and local prediction explanations.
    """

    def __init__(self, model, feature_names: list[str], tree_model=None):
        self.model = model
        self.feature_names = feature_names
        self.tree_model = tree_model if tree_model is not None else model
        self.explainer = None
        self._init_explainer()

    def _init_explainer(self):
        try:
            # Use TreeExplainer for tree models
            self.explainer = shap.TreeExplainer(self.tree_model)
        except Exception:
            # Fallback to general Explainer
            self.explainer = shap.Explainer(self.model)

    def generate_summary_plot(
        self,
        X_sample,
        output_path: str = "artifacts/explainability/shap_summary.png",
        max_display: int = 15
    ):
        """
        Generate and save global SHAP summary plot.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        dense_sample = X_sample.toarray() if sparse.issparse(X_sample) else np.array(X_sample)
        
        shap_values = self.explainer.shap_values(dense_sample)
        # For binary classification, handle list of arrays vs 3D array
        if isinstance(shap_values, list):
            sv = shap_values[1]  # Real class SHAP values
        elif len(shap_values.shape) == 3:
            sv = shap_values[:, :, 1]
        else:
            sv = shap_values

        plt.figure(figsize=(10, 7), dpi=300)
        shap.summary_plot(
            sv,
            dense_sample,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        plt.title("SHAP Global Feature Importance (News Credibility Impact)", fontsize=13, fontweight="bold", pad=15)
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        print(f"[Explainability] Saved SHAP summary plot to {output_path}")

    def generate_three_case_studies(
        self,
        real_vector,
        fake_vector,
        borderline_vector,
        real_text: str,
        fake_text: str,
        borderline_text: str,
        output_dir: str = "artifacts/explainability"
    ):
        """
        Generate and save 3 force plot / waterfall examples:
        1. High Credibility (Real News)
        2. Low Credibility (Fake News)
        3. Borderline / Ambiguous News
        Saves both interactive HTML and static high-resolution PNGs.
        """
        os.makedirs(output_dir, exist_ok=True)
        cases = [
            ("real", real_vector, "High Credibility (Real News)"),
            ("fake", fake_vector, "Low Credibility (Fake News)"),
            ("borderline", borderline_vector, "Borderline Credibility (Ambiguous Case)")
        ]

        expected_value = self.explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)) and len(expected_value) > 1:
            base_val = expected_value[1]
        else:
            base_val = float(expected_value)

        for case_name, vec, title in cases:
            dense_vec = vec.toarray().ravel() if sparse.issparse(vec) else np.array(vec).ravel()
            raw_sv = self.explainer.shap_values(dense_vec.reshape(1, -1))
            
            if isinstance(raw_sv, list):
                sv_case = raw_sv[1][0]
            elif len(raw_sv.shape) == 3:
                sv_case = raw_sv[0, :, 1]
            elif len(raw_sv.shape) == 2:
                sv_case = raw_sv[0]
            else:
                sv_case = raw_sv

            # 1. Interactive HTML Force Plot
            try:
                fp = shap.force_plot(
                    base_val,
                    sv_case,
                    dense_vec,
                    feature_names=self.feature_names,
                    out_names="Credibility Score"
                )
                html_path = os.path.join(output_dir, f"force_plot_{case_name}.html")
                shap.save_html(html_path, fp)
            except Exception as e:
                print(f"[Explainability] Warning generating HTML for {case_name}: {e}")

            # 2. High-Resolution Static Bar Plot of Top Contributing Features
            top_indices = np.argsort(np.abs(sv_case))[-10:]
            top_feats = [self.feature_names[i] for i in top_indices]
            top_vals = [sv_case[i] for i in top_indices]
            colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in top_vals]

            plt.figure(figsize=(9, 4.5), dpi=300)
            bars = plt.barh(top_feats, top_vals, color=colors, height=0.6)
            plt.axvline(0, color="black", linestyle="--", alpha=0.5, lw=1)
            plt.title(f"SHAP Feature Attribution: {title}", fontsize=11, fontweight="bold", pad=12)
            plt.xlabel("SHAP Impact on Credibility (+: Real / Credible, -: Fake / Suspicious)", fontsize=9)
            plt.tight_layout()
            png_path = os.path.join(output_dir, f"force_plot_{case_name}.png")
            plt.savefig(png_path)
            plt.close()
            print(f"[Explainability] Saved case study plot ({case_name}) to {png_path}")

    def explain_prediction(self, X_sample, top_n: int = 8) -> list[dict]:
        """
        Explain a single document prediction.
        Returns top positive (supports real) and negative (supports fake) features with values.
        """
        dense_vec = X_sample.toarray().ravel() if sparse.issparse(X_sample) else np.array(X_sample).ravel()
        raw_sv = self.explainer.shap_values(dense_vec.reshape(1, -1))
        
        if isinstance(raw_sv, list):
            sv_case = raw_sv[1][0]
        elif len(raw_sv.shape) == 3:
            sv_case = raw_sv[0, :, 1]
        elif len(raw_sv.shape) == 2:
            sv_case = raw_sv[0]
        else:
            sv_case = raw_sv

        # Sort features by absolute contribution
        active_indices = np.where(np.abs(sv_case) > 1e-5)[0]
        if len(active_indices) == 0:
            top_indices = np.argsort(np.abs(sv_case))[-top_n:][::-1]
        else:
            sorted_active = active_indices[np.argsort(np.abs(sv_case[active_indices]))[::-1]]
            top_indices = sorted_active[:top_n]

        explanations = []
        for idx in top_indices:
            feat_name = self.feature_names[idx]
            val = float(sv_case[idx])
            feat_val = float(dense_vec[idx])
            
            # Format human-friendly description
            clean_name = feat_name
            if feat_name.startswith("tfidf__"):
                clean_name = f"Term: '{feat_name.replace('tfidf__', '')}'"
            elif feat_name.startswith("aux__"):
                clean_name = f"Linguistic Metric: {feat_name.replace('aux__', '')}"
            elif feat_name.startswith("w2v__"):
                clean_name = f"Semantic Context: {feat_name.replace('w2v__', '')}"

            direction = "Supports Real" if val > 0 else "Supports Fake"
            explanations.append({
                "feature": clean_name,
                "raw_name": feat_name,
                "shap_value": round(val, 4),
                "feature_value": round(feat_val, 4),
                "direction": direction,
                "importance_percentage": round(abs(val), 4)
            })

        return explanations


if __name__ == "__main__":
    print("NewsGuard explainability module ready.")
