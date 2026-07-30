"""
Prints ONE consolidated table with the key numbers from every model you've
trained AND evaluated so far (threshold, autoencoder loss, train/val
counts, AUC/precision/recall) - no more scrolling back through a wall of
terminal output to find them.

Usage:
    python summary.py                  # every category/method found in models/
    python summary.py --category wood  # just one category
"""

import os
import json
import glob
import argparse
import numpy as np

from config import MODEL_DIR, RESULTS_DIR, CATEGORY


def load_all_meta(category_filter=None):
    rows = []
    for meta_path in sorted(glob.glob(os.path.join(MODEL_DIR, "*_meta.json"))):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            if "method" not in meta:
                raise KeyError("missing 'method' key")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"(skipping {meta_path}: {e} - probably still being written, or from an old run; "
                  f"re-run train.py for that category/method if it's missing below)")
            continue

        filename = os.path.basename(meta_path)
        category = filename.replace(f"_{meta['method']}_meta.json", "")
        if category_filter and category != category_filter:
            continue
        meta["category"] = category

        # Pull AUC/precision/recall from evaluate.py's output, if it's been run.
        eval_path = os.path.join(RESULTS_DIR, f"{category}_{meta['method']}_eval.npz")
        if os.path.exists(eval_path):
            try:
                eval_data = np.load(eval_path, allow_pickle=True)
                meta["auc"] = float(eval_data["auc"])
                meta["precision"] = float(eval_data["precision"])
                meta["recall"] = float(eval_data["recall"])
            except Exception:
                pass  # leave auc/precision/recall out - table will show "-" for this row

        rows.append(meta)
    return rows


def main(category_filter):
    rows = load_all_meta(category_filter)
    if not rows:
        print("No trained models found yet - run train.py (or run_all.py) first.")
        return

    header = (f"{'Category':<14}{'Method':<13}{'Threshold':>12}{'Final loss':>13}"
              f"{'AUC':>8}{'Precision':>11}{'Recall':>9}{'Train/Val':>12}")
    print(header)
    print("-" * len(header))
    for m in sorted(rows, key=lambda r: (r["category"], r["method"])):
        threshold = f"{m.get('threshold', float('nan')):.6f}"
        loss = f"{m['final_loss']:.6f}" if "final_loss" in m else "-"
        auc = f"{m['auc']:.3f}" if "auc" in m else "-"
        precision = f"{m['precision']:.3f}" if "precision" in m else "-"
        recall = f"{m['recall']:.3f}" if "recall" in m else "-"
        split = f"{m.get('n_train', '?')}/{m.get('n_val', '?')}"
        print(f"{m['category']:<14}{m['method']:<13}{threshold:>12}{loss:>13}"
              f"{auc:>8}{precision:>11}{recall:>9}{split:>12}")

    if any("auc" not in m for m in rows):
        print("\n(rows showing '-' for AUC/Precision/Recall haven't been evaluated yet - "
              "run evaluate.py for that category/method)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default=None,
                         help="Only show this category (default: show every category found)")
    args = parser.parse_args()
    main(args.category)
