"""
Step 5 (optional): Break results down by SPECIFIC defect type, not just
normal vs. defective. Answers: "of the crack images, how many did we catch?
Of the hole images? etc."

Usage:
    python breakdown_by_defect.py --category bottle --method cnn
    python breakdown_by_defect.py --category hazelnut --method pca

Requires that you already ran train.py + evaluate.py for this
category/method (evaluate.py must be the updated version that saves
defect_types - re-run evaluate.py if you get a KeyError about it).
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import RESULTS_DIR, CATEGORY


def main(category, method):
    path = os.path.join(RESULTS_DIR, f"{category}_{method}_eval.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Run:\n"
            f"  python train.py --category {category} --method {method}\n"
            f"  python evaluate.py --category {category} --method {method}"
        )
    data = np.load(path, allow_pickle=True)

    if "defect_types" not in data:
        raise KeyError(
            "This eval file doesn't have defect_types saved (it was created "
            "with an older version of evaluate.py). Re-run evaluate.py for "
            f"this category/method:\n  python evaluate.py --category {category} --method {method}"
        )

    defect_types = data["defect_types"]
    predictions = data["predictions"]

    print(f"Category: {category}  Method: {method}")
    print(f"{'Defect type':<20}{'Count':>8}{'Flagged':>10}{'Detection rate':>16}")
    print("-" * 54)

    rows = []
    for defect_type in sorted(set(defect_types)):
        mask = defect_types == defect_type
        count = int(mask.sum())
        flagged = int(predictions[mask].sum())
        rate = flagged / count if count > 0 else 0.0
        # For "good" (normal), "flagged" means false alarm, not a detection -
        # print it, but don't call it a "detection rate" (label it separately).
        label = "false alarm rate" if defect_type == "good" else "detection rate"
        print(f"{defect_type:<20}{count:>8}{flagged:>10}{rate:>15.1%}   ({label})")
        rows.append((defect_type, count, flagged, rate))

    # Bar chart: detection rate per defect type (good = false alarm rate, shown in a different color)
    plt.figure(figsize=(8, 4.5))
    labels_plot = [r[0] for r in rows]
    rates_plot = [r[3] for r in rows]
    colors = ["tab:red" if lbl == "good" else "tab:green" for lbl in labels_plot]
    plt.bar(labels_plot, rates_plot, color=colors)
    plt.ylim(0, 1.05)
    plt.ylabel("Detection rate (good = false alarm rate)")
    plt.title(f"{category} ({method}) - detection rate by defect type")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, f"{category}_{method}_by_defect_type.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default=None, help="Override config.CATEGORY")
    parser.add_argument("--method", choices=["pca", "autoencoder", "cnn"], default="pca")
    args = parser.parse_args()
    main(args.category or CATEGORY, args.method)
