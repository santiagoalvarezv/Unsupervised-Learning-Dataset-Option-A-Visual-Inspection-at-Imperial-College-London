"""
Step 4: Compare 2 or 3 methods for one category (run after evaluating them).

Usage:
    python compare.py --category bottle                        # pca vs autoencoder (default)
    python compare.py --category bottle --methods pca,cnn       # pca vs cnn
    python compare.py --category bottle --methods pca,autoencoder,cnn   # all three

Requires that you already ran, for each method you want to compare:
    python train.py --category bottle --method <method>
    python evaluate.py --category bottle --method <method>
(or just: python run_all.py --category bottle)
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from config import RESULTS_DIR, CATEGORY

DISPLAY_NAMES = {"pca": "PCA", "autoencoder": "Autoencoder", "cnn": "CNN features (ResNet18+PCA)"}
COLORS = {"pca": "tab:blue", "autoencoder": "tab:orange", "cnn": "tab:green"}


def load_eval(category, method):
    path = os.path.join(RESULTS_DIR, f"{category}_{method}_eval.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Run:\n"
            f"  python train.py --category {category} --method {method}\n"
            f"  python evaluate.py --category {category} --method {method}"
        )
    return np.load(path, allow_pickle=True)


def main(category, methods):
    data_by_method = {m: load_eval(category, m) for m in methods}

    print(f"Category: {category}")
    col_width = max(24, max(len(DISPLAY_NAMES[m]) for m in methods) + 2)
    header = "Metric".ljust(12) + "".join(DISPLAY_NAMES[m].rjust(col_width) for m in methods)
    print(header)
    print("-" * len(header))
    for metric in ["auc", "precision", "recall"]:
        row = metric.ljust(12) + "".join(f"{float(data_by_method[m][metric]):>{col_width}.3f}" for m in methods)
        print(row)

    best_method = max(methods, key=lambda m: float(data_by_method[m]["auc"]))
    print(f"\nHighest AUC-ROC: {DISPLAY_NAMES[best_method]}")

    # Combined ROC curve plot - the clearest single picture for "which is better"
    plt.figure(figsize=(6, 6))
    for method in methods:
        data = data_by_method[method]
        fpr, tpr, _ = roc_curve(data["labels"], data["errors"])
        plt.plot(fpr, tpr, label=f"{DISPLAY_NAMES[method]} (AUC={float(data['auc']):.3f})",
                 color=COLORS.get(method))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve comparison - {category}")
    plt.legend()
    plt.tight_layout()
    suffix = "_vs_".join(methods)
    out_path = os.path.join(RESULTS_DIR, f"{category}_{suffix}_roc.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default=None, help="Override config.CATEGORY")
    parser.add_argument("--methods", default="pca,autoencoder",
                         help="Comma-separated list of methods to compare, e.g. pca,autoencoder,cnn")
    args = parser.parse_args()
    main(args.category or CATEGORY, [m.strip() for m in args.methods.split(",")])
