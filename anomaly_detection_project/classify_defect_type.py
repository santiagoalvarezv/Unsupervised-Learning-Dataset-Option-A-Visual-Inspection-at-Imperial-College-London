"""
Real defect-TYPE classification - predicts WHICH defect an image has, not
just normal-vs-defective. Unlike breakdown_by_defect.py (which only groups
the anomaly detector's binary decision by folder name), this trains an
actual multi-class classifier on the image's visual content.

Folder names are used ONLY as labels to train against and to score
accuracy - exactly like any supervised learning problem (you need to know
the right answer to check if the model got it right). The model itself
never receives the folder name as input - only the image's deep features.

Because there's no separate "defect training set" (train/ only ever has
normal images, by design), this uses cross-validation on the test set:
each prediction is made on an image held OUT of its own training fold, so
no prediction ever "sees" its own true label beforehand.

Usage:
    python classify_defect_type.py --category hazelnut
    python classify_defect_type.py --category bottle --no-pretrained   # offline testing only
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from config import DATA_ROOT, RESULTS_DIR, CATEGORY
from data_loader import load_test_set


def main(category, pretrained):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    from cnn_model import extract_features

    # img_size/grayscale here don't matter for the CNN feature extractor
    # (it always resizes to 224x224 RGB internally) - only paths/labels do.
    _, paths, labels, defect_types = load_test_set(DATA_ROOT, category, (64, 64), True)

    # Only classify actual defects - "good" isn't a defect type to predict.
    defect_mask = labels == 1
    defect_paths = [p for p, m in zip(paths, defect_mask) if m]
    y = defect_types[defect_mask]

    if len(set(y)) < 2:
        raise RuntimeError(f"Need at least 2 different defect types to classify - found: {set(y)}")

    print(f"[{category}] Extracting ResNet18 features for {len(defect_paths)} defect images "
          f"across {len(set(y))} defect types...")
    X = extract_features(defect_paths, pretrained=pretrained)

    counts = {label: int((y == label).sum()) for label in sorted(set(y))}
    print(f"[{category}] Images per defect type: {counts}")

    min_count = min(counts.values())
    clf = RandomForestClassifier(n_estimators=200, random_state=0)

    if min_count >= 2:
        n_splits = min(5, min_count)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        cv_desc = f"{n_splits}-fold stratified cross-validation"
    else:
        cv = LeaveOneOut()
        cv_desc = "leave-one-out cross-validation (some defect types have only 1 example)"

    print(f"[{category}] Using {cv_desc}. The folder name is used ONLY to grade each "
          f"prediction after the fact - the model only ever sees image features.")

    y_pred = cross_val_predict(clf, X, y, cv=cv)

    print(f"\nCategory: {category}")
    print(classification_report(y, y_pred, zero_division=0))

    labels_sorted = sorted(set(y))
    cm = confusion_matrix(y, y_pred, labels=labels_sorted)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_sorted)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=30)
    plt.title(f"{category} - defect type classification\n({cv_desc})")
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, f"{category}_defect_type_confusion_matrix.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default=None, help="Override config.CATEGORY")
    parser.add_argument("--no-pretrained", action="store_true",
                         help="Use random-weight ResNet18 instead of ImageNet weights "
                              "(offline testing only - much weaker results)")
    args = parser.parse_args()
    main(args.category or CATEGORY, pretrained=not args.no_pretrained)
