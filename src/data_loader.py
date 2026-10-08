"""
Data Layer: Ingestion, Class Balance Analysis, Stratified 80/10/10 Split, and EDA.
Supports Fake & Real News benchmark dataset and LIAR dataset.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


def load_news_dataset(csv_path: str = "data/raw/fake_or_real_news.csv") -> pd.DataFrame:
    """
    Load the benchmark fake or real news dataset.
    Returns DataFrame with columns: ['id', 'title', 'text', 'content', 'label', 'label_name']
    where label 1 = REAL, label 0 = FAKE.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at {csv_path}")

    df = pd.read_csv(csv_path)
    
    # Handle column names
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "id"})
    elif "id" not in df.columns:
        df["id"] = range(len(df))

    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    
    # Create combined full content
    df["content"] = df["title"] + " " + df["text"]
    df["content"] = df["content"].str.strip()

    # Drop any completely empty content
    df = df[df["content"].str.len() > 10].copy()

    # Normalize labels
    if "label" in df.columns:
        df["label_name"] = df["label"].astype(str).str.upper().str.strip()
        df["label"] = df["label_name"].map({"REAL": 1, "FAKE": 0})
    else:
        raise ValueError("Dataset does not contain a 'label' column.")

    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    
    return df.reset_index(drop=True)


def load_liar_dataset(liar_dir: str = "data/raw/liar") -> pd.DataFrame:
    """
    Load and preprocess the LIAR benchmark dataset.
    Maps fine-grained politifact labels to binary:
      Real (1): true, mostly-true, half-true
      Fake (0): false, barely-true, pants-fire
    """
    splits = []
    col_names = [
        "id", "raw_label", "statement", "subject", "speaker",
        "job", "state", "party", "barely_true_c", "false_c",
        "half_true_c", "mostly_true_c", "pants_on_fire_c", "context"
    ]
    for split_file in ["train.tsv", "valid.tsv", "test.tsv"]:
        path = os.path.join(liar_dir, split_file)
        if os.path.exists(path):
            sdf = pd.read_csv(path, sep="\t", header=None, names=col_names)
            splits.append(sdf)

    if not splits:
        raise FileNotFoundError(f"No LIAR dataset files found in {liar_dir}")

    df = pd.concat(splits, ignore_index=True)
    df["statement"] = df["statement"].fillna("").astype(str)
    df["context"] = df["context"].fillna("").astype(str)
    df["content"] = df["statement"] + " Context: " + df["context"]

    # Binary label mapping
    label_map = {
        "pants-fire": 0,
        "false": 0,
        "barely-true": 0,
        "half-true": 1,
        "mostly-true": 1,
        "true": 1,
    }
    df["label"] = df["raw_label"].map(label_map)
    df["label_name"] = df["label"].map({1: "REAL", 0: "FAKE"})
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


def analyze_class_balance(df: pd.DataFrame) -> dict:
    """
    Analyze class distribution and return metrics dictionary.
    """
    counts = df["label"].value_counts().to_dict()
    total = len(df)
    n_real = counts.get(1, 0)
    n_fake = counts.get(0, 0)
    
    balance_metrics = {
        "total_samples": int(total),
        "real_count": int(n_real),
        "fake_count": int(n_fake),
        "real_percentage": round((n_real / total) * 100, 2),
        "fake_percentage": round((n_fake / total) * 100, 2),
        "imbalance_ratio": round(max(n_real, n_fake) / max(min(n_real, n_fake), 1), 3),
        "is_balanced": abs(n_real - n_fake) / total < 0.1
    }
    return balance_metrics


def perform_stratified_split(
    df: pd.DataFrame,
    train_size: float = 0.80,
    val_size: float = 0.10,
    test_size: float = 0.10,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform stratified 80/10/10 split ensuring identical class proportions
    across train, validation, and test sets with zero data leakage.
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-5, "Splits must sum to 1.0"

    # Step 1: Split train vs temp (val + test)
    temp_size = val_size + test_size
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_size,
        stratify=df["label"],
        random_state=random_state
    )

    # Step 2: Split temp into val and test (equal 50% of remaining 20% -> 10% each)
    val_ratio = val_size / temp_size
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_ratio),
        stratify=temp_df["label"],
        random_state=random_state
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, val_df, test_df


def save_processed_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str = "data/processed"
) -> dict:
    """
    Save stratified split datasets to CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    return {
        "train_path": train_path,
        "val_path": val_path,
        "test_path": test_path,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df)
    }


def generate_eda(df: pd.DataFrame, figures_dir: str = "artifacts/figures", metrics_dir: str = "artifacts/metrics") -> dict:
    """
    Run Exploratory Data Analysis (EDA) on text data and save figures and summary statistics.
    """
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Calculate basic text statistics
    df_eda = df.copy()
    df_eda["char_count"] = df_eda["content"].str.len()
    df_eda["word_count"] = df_eda["content"].apply(lambda x: len(x.split()))
    df_eda["avg_word_len"] = df_eda["char_count"] / df_eda["word_count"].replace(0, 1)

    summary_stats = {
        "class_balance": analyze_class_balance(df_eda),
        "word_count_stats": {
            "overall_mean": float(df_eda["word_count"].mean()),
            "overall_median": float(df_eda["word_count"].median()),
            "real_mean": float(df_eda[df_eda["label"] == 1]["word_count"].mean()),
            "fake_mean": float(df_eda[df_eda["label"] == 0]["word_count"].mean()),
            "real_median": float(df_eda[df_eda["label"] == 1]["word_count"].median()),
            "fake_median": float(df_eda[df_eda["label"] == 0]["word_count"].median()),
        },
        "char_count_stats": {
            "overall_mean": float(df_eda["char_count"].mean()),
            "real_mean": float(df_eda[df_eda["label"] == 1]["char_count"].mean()),
            "fake_mean": float(df_eda[df_eda["label"] == 0]["char_count"].mean()),
        }
    }

    # Save summary stats JSON
    summary_path = os.path.join(metrics_dir, "eda_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2)

    # Figure 1: Class Distribution
    plt.figure(figsize=(7, 5), dpi=300)
    palette = ["#e74c3c", "#2ecc71"]
    counts = df_eda["label_name"].value_counts()
    ax = sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=palette, legend=False)
    plt.title("NewsGuard Dataset: Class Distribution (Real vs Fake)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Article Class", fontsize=11)
    plt.ylabel("Number of Articles", fontsize=11)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())} ({p.get_height()/len(df_eda):.1%})",
                    (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                    ha="center", va="center", color="white", fontweight="bold", fontsize=11)
    plt.tight_layout()
    class_fig_path = os.path.join(figures_dir, "eda_class_distribution.png")
    plt.savefig(class_fig_path)
    plt.close()

    # Figure 2: Article Length Distribution (Word Count)
    plt.figure(figsize=(9, 5), dpi=300)
    clip_words = df_eda["word_count"].clip(upper=2500)
    sns.kdeplot(clip_words[df_eda["label"] == 1], label="Real News", color="#2ecc71", fill=True, alpha=0.35, linewidth=2)
    sns.kdeplot(clip_words[df_eda["label"] == 0], label="Fake News", color="#e74c3c", fill=True, alpha=0.35, linewidth=2)
    plt.title("Article Word Count Distribution by Credibility Label", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Word Count (clipped at 2500 words for visualization)", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.legend(frameon=True, facecolor="white", loc="upper right")
    plt.tight_layout()
    len_fig_path = os.path.join(figures_dir, "eda_length_distribution.png")
    plt.savefig(len_fig_path)
    plt.close()

    print(f"[EDA] Generated class distribution figure: {class_fig_path}")
    print(f"[EDA] Generated length distribution figure: {len_fig_path}")
    print(f"[EDA] Saved summary metrics: {summary_path}")

    return summary_stats


def prepare_data_pipeline(dataset_type: str = "news") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Main entry point for Day 1: Loads dataset, performs EDA, and creates 80/10/10 stratified split.
    """
    if dataset_type == "news":
        df = load_news_dataset()
    elif dataset_type == "liar":
        df = load_liar_dataset()
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    print(f"[Data Loader] Loaded {len(df)} samples from {dataset_type} dataset.")
    eda_stats = generate_eda(df)
    train_df, val_df, test_df = perform_stratified_split(df)
    split_info = save_processed_splits(train_df, val_df, test_df)

    print(f"[Data Loader] Split completed: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df, split_info


if __name__ == "__main__":
    train_df, val_df, test_df, info = prepare_data_pipeline()
    print("Dataset preparation successful:", info)
