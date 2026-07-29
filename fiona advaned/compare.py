"""Compare any two or three retained methods for one product category."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from config import CATEGORY, MODEL_DIR, RESULTS_DIR

DISPLAY_NAMES = {
    "pca": "Raw-pixel PCA",
    "autoencoder": "Basic ConvAE",
    "cnn": "ResNet18 features + PCA",
}


def load_eval(category: str, method: str):
    path = os.path.join(RESULTS_DIR, f"{category}_{method}_eval.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Train and evaluate {method} first.")
    return np.load(path, allow_pickle=True)


def load_meta(category: str, method: str) -> dict:
    path = os.path.join(MODEL_DIR, f"{category}_{method}_meta.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}.")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_shared_split(category: str, methods: list[str]):
    metadata = {method: load_meta(category, method) for method in methods}
    reference = metadata[methods[0]]
    for method in methods[1:]:
        current = metadata[method]
        for key in ["fit_paths", "validation_paths", "validation_fraction", "random_seed"]:
            if current.get(key) != reference.get(key):
                raise RuntimeError(f"Methods do not share the same normal split: mismatch in {key}.")

    if "pca" in methods and "autoencoder" in methods:
        pca_meta = metadata["pca"]
        ae_meta = metadata["autoencoder"]
        for key in ["img_size", "grayscale", "threshold_percentile"]:
            if pca_meta.get(key) != ae_meta.get(key):
                raise RuntimeError(f"PCA and ConvAE comparison is not fair: mismatch in {key}.")
        if pca_meta.get("pca_components") != ae_meta.get("latent_dim"):
            print(
                "Warning: PCA component count and ConvAE latent dimension differ: "
                f"{pca_meta.get('pca_components')} vs {ae_meta.get('latent_dim')}."
            )


def scalar(data, key: str) -> float:
    return float(data[key]) if key in data.files else float("nan")


def main(category: str, methods: list[str]):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    verify_shared_split(category, methods)
    data_by_method = {method: load_eval(category, method) for method in methods}

    metric_keys = ["auc", "average_precision", "precision", "recall", "f1", "specificity"]
    print(f"Category: {category}")
    width = max(24, max(len(DISPLAY_NAMES[method]) for method in methods) + 2)
    header = "Metric".ljust(20) + "".join(DISPLAY_NAMES[m].rjust(width) for m in methods)
    print(header)
    print("-" * len(header))
    for metric in metric_keys:
        row = metric.ljust(20) + "".join(f"{scalar(data_by_method[m], metric):>{width}.3f}" for m in methods)
        print(row)

    output_csv = os.path.join(RESULTS_DIR, f"{category}_{'_vs_'.join(methods)}_metrics.csv")
    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", *metric_keys, "tn", "fp", "fn", "tp"])
        for method in methods:
            data = data_by_method[method]
            writer.writerow(
                [DISPLAY_NAMES[method]]
                + [scalar(data, key) for key in metric_keys]
                + [int(data[key]) for key in ["tn", "fp", "fn", "tp"]]
            )

    plt.figure(figsize=(6, 6))
    for method in methods:
        data = data_by_method[method]
        fpr, tpr, _ = roc_curve(data["labels"], data["errors"])
        plt.plot(fpr, tpr, label=f"{DISPLAY_NAMES[method]} (AUC={scalar(data, 'auc'):.3f})")
    plt.plot([0, 1], [0, 1], "--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC comparison - {category}")
    plt.legend()
    plt.tight_layout()
    roc_path = os.path.join(RESULTS_DIR, f"{category}_{'_vs_'.join(methods)}_roc.png")
    plt.savefig(roc_path, dpi=180)
    plt.close()

    plt.figure(figsize=(6, 6))
    for method in methods:
        data = data_by_method[method]
        precision, recall, _ = precision_recall_curve(data["labels"], data["errors"])
        plt.plot(
            recall,
            precision,
            label=f"{DISPLAY_NAMES[method]} (AP={scalar(data, 'average_precision'):.3f})",
        )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-recall comparison - {category}")
    plt.legend()
    plt.tight_layout()
    pr_path = os.path.join(RESULTS_DIR, f"{category}_{'_vs_'.join(methods)}_pr.png")
    plt.savefig(pr_path, dpi=180)
    plt.close()

    best = max(methods, key=lambda method: scalar(data_by_method[method], "auc"))
    print(f"Highest AUC-ROC: {DISPLAY_NAMES[best]}")
    print(f"Saved {output_csv}")
    print(f"Saved {roc_path}")
    print(f"Saved {pr_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default=CATEGORY)
    parser.add_argument("--methods", default="pca,autoencoder")
    args = parser.parse_args()
    selected = [method.strip() for method in args.methods.split(",") if method.strip()]
    allowed = {"pca", "autoencoder", "cnn"}
    if not selected or any(method not in allowed for method in selected):
        raise SystemExit("--methods must contain pca, autoencoder, and/or cnn.")
    main(args.category, selected)
